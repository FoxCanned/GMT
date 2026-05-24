import cv2
import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import numpy as np

from detectron2.config import configurable
from detectron2.structures import Boxes, pairwise_iou, Instances

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from .custom_rcnn import CustomRCNN
from ..roi_heads.custom_fast_rcnn import custom_fast_rcnn_inference
from tqdm import tqdm
import time
import copy
from detectron2.structures import ImageList, Instances


class poss_ids:
    poss_ids = set()

    def __init__(self):
        pass

class old_ids:
    old_ids = set()
    def __init__(self):
        pass
    
class old_reids:
    old_reids = []
    def __init__(self):
        pass

@META_ARCH_REGISTRY.register()
class GTRRCNN(CustomRCNN):
    @configurable
    def __init__(self, **kwargs):
        """
        """
        self.test_len = kwargs.pop('test_len')
        self.overlap_thresh = kwargs.pop('overlap_thresh')
        self.min_track_len = kwargs.pop('min_track_len')
        self.max_center_dist = kwargs.pop('max_center_dist')
        self.decay_time = kwargs.pop('decay_time')
        self.asso_thresh = kwargs.pop('asso_thresh')
        self.with_iou = kwargs.pop('with_iou')
        self.local_track = kwargs.pop('local_track')
        self.local_no_iou = kwargs.pop('local_no_iou')
        self.local_iou_only = kwargs.pop('local_iou_only')
        self.not_mult_thresh = kwargs.pop('not_mult_thresh')
        self.with_attention = kwargs.pop('with_attention')
        self.thred_bank = kwargs.pop('thred')
        self.bank_size = kwargs.pop('bank_size')
        self.with_bank = kwargs.pop('with_bank')
        self.multi_modal = kwargs.pop('multi_modal')
        super().__init__(**kwargs)


    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['test_len'] = cfg.INPUT.VIDEO.TEST_LEN
        ret['overlap_thresh'] = cfg.VIDEO_TEST.OVERLAP_THRESH     
        ret['asso_thresh'] = cfg.MODEL.ASSO_HEAD.ASSO_THRESH
        ret['min_track_len'] = cfg.VIDEO_TEST.MIN_TRACK_LEN
        ret['max_center_dist'] = cfg.VIDEO_TEST.MAX_CENTER_DIST
        ret['decay_time'] = cfg.VIDEO_TEST.DECAY_TIME
        ret['with_iou'] = cfg.VIDEO_TEST.WITH_IOU
        ret['local_track'] = cfg.VIDEO_TEST.LOCAL_TRACK
        ret['local_no_iou'] = cfg.VIDEO_TEST.LOCAL_NO_IOU
        ret['local_iou_only'] = cfg.VIDEO_TEST.LOCAL_IOU_ONLY
        ret['not_mult_thresh'] = cfg.VIDEO_TEST.NOT_MULT_THRESH
        ret['with_attention'] = cfg.MODEL.ASSO_HEAD.WITH_ATTENTION
        ret['thred'] = cfg.MODEL.ASSO_HEAD.THRED
        ret['bank_size'] = cfg.MODEL.ASSO_HEAD.BANK_SIZE
        ret['with_bank'] = cfg.MODEL.ASSO_HEAD.WITH_BANK
        ret['multi_modal'] = cfg.MULTI_MODAL
        return ret


    def forward(self, batched_inputs):
        """
        All batched images are from the same video
        During testing, the current implementation requires all frames 
            in a video are loaded.
        TODO (Xingyi): one-the-fly testing
        """
        min = 999999
        max = -99999
        frame_num = []
        for num,i in enumerate(batched_inputs):
            num = int(i['file_name'].split('.')[0][-6:])

            #mvmhat
            #num = int(i['file_name'].split('.')[0].split('/')[-1])
            if num>max:
                max = num
            if num<min:
                min = num
            frame_num.append(num)
        view_num = batched_inputs[0]['view_num']
        if not self.training:
            if self.local_track:
                return self.local_tracker_inference(batched_inputs)
            return self.sliding_inference_GMT(batched_inputs, view_num, [min, max, frame_num])

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs ]
        if self.multi_modal:
            features = self.fuse_feature(features)
            gt_instances = [x["instances"].to(self.device) for i,x in enumerate(batched_inputs) if i%2==0 ]
            images = ImageList(images.tensor[range(0,len(images),2)],images.image_sizes[0:len(images):2])
        try:
            view_num = batched_inputs[0]["view_num"]
        except :
            view_num = -1

        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)
        _, detector_losses = self.roi_heads(
            images, features, proposals,view_num,[min,max,frame_num],None, gt_instances)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        return losses

               

    def sliding_inference(self, batched_inputs):
        video_len = len(batched_inputs)
        instances = []
        id_count = 0
        for frame_id in range(video_len):
            instances_wo_id = self.inference(
                batched_inputs[frame_id: frame_id + 1], 
                do_postprocess=False)
            instances.extend([x for x in instances_wo_id])

            if frame_id == 0: # first frame
                instances[0].track_ids = torch.arange(
                    1, len(instances[0]) + 1,
                    device=instances[0].reid_features.device)
                id_count = len(instances[0]) + 1
            else:
                win_st = max(0, frame_id + 1 - self.test_len)
                win_ed = frame_id + 1
                instances[win_st: win_ed], id_count = self.run_global_tracker(
                    batched_inputs[win_st: win_ed],
                    instances[win_st: win_ed],
                    k=min(self.test_len - 1, frame_id),
                    id_count=id_count) # n_k x N
            if frame_id - self.test_len >= 0:
                instances[frame_id - self.test_len].remove(
                    'reid_features')

        if self.min_track_len > 0:
            instances = self._remove_short_track(instances)
        if self.roi_heads.delay_cls:
            instances = self._delay_cls(
                instances, video_id=batched_inputs[0]['video_id'])
        instances = CustomRCNN._postprocess(
                instances, batched_inputs, [
                    (0, 0) for _ in range(len(batched_inputs))],
                not_clamp_box=self.not_clamp_box)
        return instances
    
    def sliding_inference_GMT(self, batched_inputs,view_num,time):
        poss_ids.poss_ids = set()
        old_ids.old_ids = set()
        old_reids.old_reids = []
        view_num = batched_inputs[0]['view_num']
        view_frames = int(len(batched_inputs)/view_num)
        instances = []
        id_count = 0
        id_count_dict = dict()
        id_reid_dict = dict()
        memory_bank = []
        for frame_id in tqdm(range(view_frames)):
            batched_inputs_divo = []
            st = frame_id
            instances_wo_id = []
            time_per = time.copy()
            for view in range(view_num):
                time_per[2] = time[2][st]
                instances_wo_id += self.inference(
                    batched_inputs[st: st + 1],
                    view_num,
                    time_per,
                    view,
                    do_postprocess=False)
                st += view_frames
            instances.extend([x for x in instances_wo_id])
            activate_first = True
            activate = True

            if frame_id == 0:
                first_frame_num = [len(instances[i])  for i in range(view_num)]
                sort_index = np.argsort(first_frame_num)
                sort_index = sort_index.tolist()
                sort_index.reverse()
                max_index = sort_index[0]

                instances[max_index].track_ids = torch.arange(
                    1, len(instances[max_index]) + 1,
                    device=instances[max_index].reid_features.device)
                id_count = len(instances[max_index]) 
                for i in range(1, len(instances[max_index]) + 1):
                    id_count_dict[i] = 1
                    id_reid_dict[i] = instances[max_index][i-1]
                #id = ([i for i in range(view_num)])
                sort_index.remove(max_index)
                id = np.sort(sort_index)
                instances_kv = [instances[max_index]]
                if activate_first :
                    x = []

                    for i in range(view_num-1):
                        x = [instances[id[i]]]
                        a = Instances(instances[0].image_size)
                        a = a.cat(x)
                        instances_kv += [a]
                        asso_output, pred_boxes, n_t, Np, query_inds = self.get_asso(
                            instances_kv,
                            k=len(instances_kv) - 1)  # n_k x N

                        instances_kv =instances_kv
                        instances_kv, id_count,id_count_dict,id_reid_dict  = self.run_first_tracker_plus(
                            instances_kv,
                            [asso_output[0][:,:Np]],
                            pred_boxes[:Np,:],
                            len(instances_kv)-1,
                            id_count,
                            id_count_dict,
                            id_reid_dict)           
                        #start = end
                        instances[id[i]] = instances_kv[len(instances_kv)-1]
                else:
                   # instances_kv = [instances[max_index]]
                    for i in range(view_num-1):
                        instances_kv = instances_kv + [instances[id[i]]]
                        #instances_kv = instances_kv + [instances[i+1]]
                        instances_kv, id_count = self.run_first_tracker(
                            batched_inputs[: 2],
                            instances_kv,
                            k=len(instances_kv)-1,
                            id_count=id_count) # n_k x N
                        instances[id[i]] = instances_kv[len(instances_kv)-1]
                        #query += 1
                    id_count = 0
                    for i in range(view_num):
                        id_count = max(id_count,(max(instances[i].track_ids )).item())
            else:
                win_st = max(0, frame_id + 1 - self.test_len)*view_num
                win_ed = view_num*frame_id
               # activate  = True
                instances_kv = instances[win_st:win_ed]
                if  activate:
                    instacnes_old = instances[:win_st]
                    for i in range(view_num):
                        instances_kv = instances_kv + [instances[win_ed+i]]
                        asso_output, pred_boxes, n_t, Np, query_inds = self.get_asso(
                            instances_kv,
                            k=len(instances_kv) - 1)

                        instances_kv, id_count,id_count_dict = self.run_global_tracker_plus(
                            view_num,
                            instances_kv,
                            [asso_output[0][:,:Np]],
                            pred_boxes[:Np,:],
                            len(instances_kv)-1,
                            id_count,
                            id_count_dict,
                            id_reid_dict,
                            instacnes_old,
                            i)
                        instances[win_ed+i] = instances_kv[-1]
                else :
                    for i in range(view_num):
                        instances_kv =instances_kv + [instances[win_ed+i]]
                        instances_kv, id_count = self.run_global_tracker(
                            batched_inputs[win_st: win_ed],
                            instances_kv,
                            k=len(instances_kv)-1,
                            id_count=id_count) # n_k x N
                        instances[win_ed+i] = instances_kv[len(instances_kv)-1]


        batch = []
        #调整batch的顺序，和instances一致，view1_frame1,view_2_frame1,view_3_frame1 
        for i in range(view_frames):
            for j in range(view_num):
                batch.append(batched_inputs[j*view_frames+i])
        if self.min_track_len > 0:
            instances = self._remove_short_track(instances)
        if self.roi_heads.delay_cls:
            instances = self._delay_cls(
                instances, video_id=batched_inputs[0]['video_id'])
        instances = CustomRCNN._postprocess(
                instances, batch, [
                    (0, 0) for _ in range(len(batch))],
                not_clamp_box=self.not_clamp_box)
        for i in range(len(batch)):
            batch[i]['image'] = None        
        return instances,view_num

    def run_first_tracker_plus(self, instances,asso_output,pred_boxes,k,id_count,id_count_dict,id_reid_dict):
        n_t = [len(x) for x in instances]
        N, T = sum(n_t), len(n_t)
        asso_nonk = self.roi_heads._activate_asso(asso_output)[0]

        n_k = len(instances[k])
        Np = N - n_k
        ids = torch.cat(
            [x.track_ids for t, x in enumerate(instances) if t != k],
            dim=0).view(Np) # Np

        unique_ids = torch.unique(ids) # M
        id_inds = (unique_ids[None, :] == ids[:, None]).float() # Np x M

        traj_score = torch.mm(asso_nonk, id_inds) # n_k x M

        match_i, match_j = linear_sum_assignment((- traj_score).cpu()) #
        track_ids = ids.new_full((n_k,), -1)
        for i, j in zip(match_i, match_j):
            thresh = self.overlap_thresh * id_inds[:, j].sum() \
                if not (self.not_mult_thresh) else self.overlap_thresh
            if traj_score[i, j] > thresh:
                track_ids[i] = unique_ids[j]

        for i in range(n_k):
            id =  track_ids[i].item()
            if track_ids[i] < 0:
                id_count = id_count + 1
                track_ids[i] = id_count
                id_count_dict[id_count] = 1
                instances[k].track_ids = track_ids#修改
                id_reid_dict[id_count] = instances[k][i]
            else :
                id_count_dict[id] += 1
                instances[k].track_ids = track_ids#修改
                instance_cat = [id_reid_dict[id] , instances[k][i]]
                id_reid_dict[id] = Instances.cat(instance_cat)
                #id_reid_dict[id].reid_features = id_reid_dict[id].reid_features/id_count_dict[id]*(id_count_dict[id]-1)+instances[k][i].reid_features/id_count_dict[id]
        
        instances[k].track_ids = track_ids

        assert len(track_ids) == len(torch.unique(track_ids)), track_ids
        return instances, id_count ,id_count_dict,id_reid_dict

    def get_asso(self,  instances, k,):
        n_t = [len(x) for x in instances]
        N, T = sum(n_t), len(n_t)
        n_k = len(instances[k])
        Np = N - n_k
        traj_ids=torch.Tensor().cuda()
        for i in range(len(instances)-1):
            traj_ids=torch.cat([traj_ids,instances[i].track_ids])
        reid_features = torch.cat(
                [x.reid_features for x in instances], dim=0)[None]
        asso_output, pred_boxes, _, query_inds,_,_ = self.roi_heads._forward_transformer(
            instances, reid_features,None, k,None,None,traj_ids) # [n_k x N], N x 4
        return asso_output,pred_boxes,n_t,Np,query_inds

    def get_attention(self,instances,view_num,query_inds,id=None):
        proposals = [x for  x in  instances]
        #features = [features[f] for f in self.asso_in_features]
        proposal_boxes = [x.pred_boxes for x in proposals] # 
        n_t = [len(x) for x in proposals]
        if id==None:
            num_frame = int(len(instances)/view_num)
            spatial_feature = torch.zeros((1,sum(n_t),4))
            for i in range(len(n_t)):
                for j in range(4):
                    if j==0 or j==2:
                        spatial_feature[0,sum(n_t[:i]):sum(n_t[:i+1]),j] = proposal_boxes[i].tensor[:,j]/instances[i].image_size[1]
                    else :
                        spatial_feature[0,sum(n_t[:i]):sum(n_t[:i+1]),j] = proposal_boxes[i].tensor[:,j]/instances[i].image_size[0]
            camera_feature = torch.zeros((1,sum(n_t),1))
            for i in range(len(n_t)):
                camera_feature[0,sum(n_t[:i]):sum(n_t[:i+1]),0] = (i%view_num)/float(view_num)
            time_featrue = torch.zeros((1,sum(n_t),1))
            for i in range(num_frame):
                time_featrue[0,sum(n_t[:i*view_num]):sum(n_t[:(i+1)*view_num]),0] = float(i)/(num_frame)
            spatial_feature = spatial_feature.to(proposal_boxes[0].device)
            time_featrue = time_featrue.to(proposal_boxes[0].device)
            camera_feature = camera_feature.to(proposal_boxes[0].device)
            inputs=[spatial_feature,time_featrue,camera_feature]
            s_t_feature = torch.cat(inputs,dim=2)
        else :
            num_frame = int(len(instances)/view_num)
            spatial_feature = torch.zeros((1,sum(n_t),4))
            for i in range(len(n_t)):
                for j in range(4):
                    if j==0 or j==2:
                        spatial_feature[0,sum(n_t[:i]):sum(n_t[:i+1]),j] = proposal_boxes[i].tensor[:,j]/instances[i].image_size[1]
                    else :
                        spatial_feature[0,sum(n_t[:i]):sum(n_t[:i+1]),j] = proposal_boxes[i].tensor[:,j]/instances[i].image_size[0]
            camera_feature = torch.zeros((1,sum(n_t),1))
            for i in range(view_num):
                camera_feature[0,sum(n_t[0:(i)*num_frame]):sum(n_t[0:(i+1)*num_frame]),0] = i/float(view_num)
            time_featrue = torch.zeros((1,sum(n_t),1))
            for i in range(len(n_t)):
                time_featrue[0,sum(n_t[:i]):sum(n_t[:i+1]),0] = float(i%num_frame)/(num_frame)
            spatial_feature = spatial_feature.to(proposal_boxes[0].device)
            time_featrue = time_featrue.to(proposal_boxes[0].device)
            camera_feature = camera_feature.to(proposal_boxes[0].device)
            inputs=[spatial_feature,time_featrue,camera_feature]
            s_t_feature_temp = torch.cat(inputs,dim=2)
            s_t_feature = torch.zeros_like(s_t_feature_temp)
            start = 0
            for i in range(len(n_t)):
                s_t_feature[0,start:start+n_t[id[i]],:] = s_t_feature_temp[0,sum(n_t[:id[i]]) :sum(n_t[:id[i]+1]),:]
                start += n_t[id[i]]
        return    self.roi_heads._forward_transformer_attention(self.roi_heads.s_t_head(s_t_feature),query_inds)
                
    def run_global_tracker_plus(self, view_num,instances,asso_output,pred_boxes,k,id_count,id_count_dict,id_reid_dict,instances_old,view):
        n_t = [len(x) for x in instances]
        N, T = sum(n_t), len(n_t)
        #view_num = 3 #这部分代码要把3替换调
        asso_output = asso_output[-1].split(n_t[:-1], dim=1) # T x [n_k x n_t]
        asso_output = self.roi_heads._activate_asso(asso_output) # T x [n_k x n_t]
        asso_nonk = torch.cat(asso_output, dim=1) # n_k x N
        last_frame_num = 2
        n_k = len(instances[k])
        Np = N - n_k
        ids = torch.cat(
            [x.track_ids for t, x in enumerate(instances) if t != k],
            dim=0).view(Np) # Np
        nonk_inds_view = []
        nonk_inds_last = []
        start = 0
        end = 0
        for t,x in enumerate(n_t):
            end += x
            if t!=k and (k-t)%view_num ==0:
                nonk_inds_view +=   range(start,end)
            if t!=k and (k-t)%view_num ==0 and (k-t)/view_num<=last_frame_num:
                nonk_inds_last +=   range(start,end)            
            start += x


        unique_ids = torch.unique(ids) # M

        M = len(unique_ids) # number of existing tracks
        id_inds = (unique_ids[None, :] == ids[:, None]).float() # Np x M

        traj_score = torch.mm(asso_nonk, id_inds) # n_k x M

        match_i, match_j = linear_sum_assignment((- traj_score).cpu()) #
        track_ids = ids.new_full((n_k,), -1)
        for i, j in zip(match_i, match_j):
            thresh = self.overlap_thresh * id_inds[:, j].sum() \
                if not (self.not_mult_thresh) else self.overlap_thresh
            if traj_score[i, j] > thresh:
                track_ids[i] = unique_ids[j]
        if self.with_bank:
            flag = False
            #a = Instances(instances[0].image_size)
            isinstances_no_match = [instances[-1]]#   取最后一个待匹配帧
            index = []
            for i in range(n_k):
                if track_ids[i] < 0:
                    index.append(i)
            isinstances_no_match[0] = isinstances_no_match[0][index]#   取所有没有匹配上的instance
            track_id_memory ,instances_matching,run_time= self.memory_bank(id_count_dict,id_reid_dict,unique_ids,instances_old,isinstances_no_match)
            count = 0
            if track_id_memory!=None:
                for i in range(n_k):
                    if track_ids[i] < 0 :
                        if track_id_memory[count]>=0:
                            track_ids[i] = track_id_memory[count]
                           # print("************ohhhhhhhhhhhh********")
                            print('id',track_id_memory[count],'view',view)
                            print(run_time)
                        count += 1
        #poss_ids.poss_ids = set()
        for i in range(n_k):
            id = track_ids[i].item()
            if track_ids[i] < 0:
                id_count = id_count + 1
                track_ids[i] = id_count
                id_count_dict[id_count] = 1
                instances[k].track_ids = track_ids#修改
                id_reid_dict[id_count] = instances[k][i]
            else :
                id_count_dict[id] += 1
                if id_count_dict[id]==self.bank_size+1:
                    poss_ids.poss_ids.add(id)
                instances[k].track_ids = track_ids#修改
                instance_cat = [id_reid_dict[id] , instances[k][i]]
                id_reid_dict[id] = Instances.cat(instance_cat)
                #id_reid_dict[id].reid_features = id_reid_dict[id].reid_features/id_count_dict[id]*(id_count_dict[id]-1)+instances[k][i].reid_features/id_count_dict[id]
        instances[k].track_ids = track_ids

        assert len(track_ids) == len(torch.unique(track_ids)), track_ids
        return instances, id_count,id_count_dict
 
    def memory_bank(self,id_count_dict,id_reid_dict,unique_ids,instances_old,instances_no_match):
        time1 = time.time()
        if len(instances_old)==0 or len(instances_no_match[0])==0:
            return  None,None,None
        instacnes_to_match = []
        thred = self.bank_size
        instances_old.reverse()
        memory_ids=[]
        for id in poss_ids.poss_ids.copy():
            sum_reid = 0
            if id not in unique_ids:
                memory_ids.append(id)
                #old_ids.old_ids.add(id)
                poss_ids.poss_ids.remove(id)
                for i in range(thred):
                    sum_reid += id_reid_dict[id][-1-i].reid_features
                aver_reid = sum_reid / thred
                #id_reid_dict[id][0].reid_features = aver_reid
                old_reids.old_reids.append(id_reid_dict[id][0])
                if len(old_reids.old_reids)==1:
                    old_reids.old_reids[0].reid_features = aver_reid
                else:
                    old_reids.old_reids[1].reid_features = aver_reid
                old_reids.old_reids = Instances.cat(old_reids.old_reids)
                old_reids.old_reids = [old_reids.old_reids]
                    
        time2 = time.time()

        if len(old_reids.old_reids)==0:
            return None,None,None

        instances_matching = old_reids.old_reids.copy()
        instances_matching.append(instances_no_match[0])
        time3 = time.time()
        asso_output,pred_boxes,n_t,Np,query_inds= self.get_asso(
            instances_matching,
            k= len(instances_matching)-1) # n_k x N       
        time4 = time.time()
        track_ids = self.run_memory_tracker(instances_matching,asso_output,pred_boxes,len(instances_matching)-1,id_count_dict)
        time5 = time.time()
        return track_ids, instacnes_to_match,[time2-time1,time3-time2,time4-time3,time5-time4]
                
#                if id in instances_old[i]['track_ids']
    def run_memory_tracker(self, instances,asso_output,pred_boxes,k,id_count_dict):    
        n_t = [len(x) for x in instances]
        N, T = sum(n_t), len(n_t)
        asso_nonk = self.roi_heads._activate_asso(asso_output)[0]

        n_k = len(instances[k])
        Np = N - n_k
        ids = torch.cat(
            [x.track_ids for t, x in enumerate(instances) if t != k],
            dim=0).view(Np) # Np

        unique_ids = torch.unique(ids) # M
        id_inds = (unique_ids[None, :] == ids[:, None]).float() # Np x M

        traj_score = torch.mm(asso_nonk, id_inds) # n_k x M

        match_i, match_j = linear_sum_assignment((- traj_score).cpu()) #
        track_ids = ids.new_full((n_k,), -1)
        for i, j in zip(match_i, match_j):
            thresh = self.thred_bank * id_inds[:, j].sum() \
                if not (self.not_mult_thresh) else self.thred_bank
            if traj_score[i, j] > thresh:
                track_ids[i] = unique_ids[j]
                #old_ids.old_ids.remove(unique_ids[j].item())
                poss_ids.poss_ids.add(unique_ids[j].item())
                a = len(old_reids.old_reids[0])
                for i in range(a):
                    if old_reids.old_reids[0][i].track_ids.item() ==unique_ids[j].item():
                        old_reids.old_reids = [Instances.cat([old_reids.old_reids[0][:i],old_reids.old_reids[0][i+1:]])]
                        break

       # assert len(track_ids) == len(torch.unique(track_ids)), track_ids
        return track_ids

    def _remove_short_track(self, instances):
        ids = torch.cat([x.track_ids for x in instances], dim=0) # N
        unique_ids = ids.unique() # M
        id_inds = (unique_ids[:, None] == ids[None, :]).float() # M x N
        num_insts_track = id_inds.sum(dim=1) # M
        remove_track_id = num_insts_track < self.min_track_len # M
        unique_ids[remove_track_id] = -1
        ids = unique_ids[torch.where(id_inds.permute(1, 0))[1]]
        ids = ids.split([len(x) for x in instances])
        for k in range(len(instances)):
            instances[k] = instances[k][ids[k] >= 0]
        return instances


    def _delay_cls(self, instances, video_id):
        ids = torch.cat([x.track_ids for x in instances], dim=0) # N
        unique_ids = ids.unique() # M
        M = len(unique_ids) # #existing tracks
        id_inds = (unique_ids[:, None] == ids[None, :]).float() # M x N
        # update scores
        cls_scores = torch.cat(
            [x.cls_scores for x in instances], dim=0) # N x (C + 1)
        traj_scores = torch.mm(id_inds, cls_scores) / \
            (id_inds.sum(dim=1)[:, None] + 1e-8) # M x (C + 1)
        _, traj_inds = torch.where(id_inds.permute(1, 0)) # N
        cls_scores = traj_scores[traj_inds] # N x (C + 1)

        n_t = [len(x) for x in instances]
        boxes = [x.pred_boxes.tensor for x in instances]
        track_ids = ids.split(n_t, dim=0)
        cls_scores = cls_scores.split(n_t, dim=0)
        instances, _ = custom_fast_rcnn_inference(
            boxes, cls_scores, track_ids, [None for _ in n_t],
            [x.image_size for x in instances],
            self.roi_heads.box_predictor[-1].test_score_thresh,
            self.roi_heads.box_predictor[-1].test_nms_thresh,
            self.roi_heads.box_predictor[-1].test_topk_per_image,
            self.not_clamp_box,
        )
        for inst in instances:
            inst.track_ids = inst.track_ids + inst.pred_classes * 10000 + \
                video_id * 100000000
        return instances

    def local_tracker_inference(self, batched_inputs):
        from ...tracking.local_tracker.fairmot import FairMOT
        local_tracker = FairMOT(
            no_iou=self.local_no_iou,
            iou_only=self.local_iou_only)

        video_len = len(batched_inputs)
        instances = []
        ret_instances = []
        for frame_id in range(video_len):
            instances_wo_id = self.inference(
                batched_inputs[frame_id: frame_id + 1], 
                do_postprocess=False)
            instances.extend([x for x in instances_wo_id])
            inst = instances[frame_id]
            dets = torch.cat([
                inst.pred_boxes.tensor, 
                inst.scores[:, None]], dim=1).cpu()
            id_feature = inst.reid_features.cpu()
            tracks = local_tracker.update(dets, id_feature)
            track_inds = [x.ind for x in tracks]
            ret_inst = inst[track_inds]
            track_ids = [x.track_id for x in tracks]
            ret_inst.track_ids = ret_inst.pred_classes.new_tensor(track_ids)
            ret_instances.append(ret_inst)
        instances = ret_instances

        if self.min_track_len > 0:
            instances = self._remove_short_track(instances)
        if self.roi_heads.delay_cls:
            instances = self._delay_cls(
                instances, video_id=batched_inputs[0]['video_id'])
        instances = CustomRCNN._postprocess(
                instances, batched_inputs, [
                    (0, 0) for _ in range(len(batched_inputs))],
                not_clamp_box=self.not_clamp_box)
        return instances
