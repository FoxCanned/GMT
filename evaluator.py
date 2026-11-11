# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn
import gc
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
import os
import cv2
import colorsys




class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results




def log_track(outputs,view_nums,scene):
    frames = int(len(outputs)/view_nums)
    name = 'PET18000112_720_25_objdetection0.525_nodecay_multithred0.001_NMS0.65'
    #name = 'test_code'
    name = 'VISIONT18000_13_640_60_objdetection0.525_multithred0.001_NMS0.65_MINLEN50'    
    root = './'
    result_root = root+name
    seqs_dict = {'circleRegion': 'Circle',
                'innerShop': 'Shop',
                'movingView': 'Moving',
                'park': 'Park',
                'playground': 'Ground',
                'shopFrontGate': 'Gate1',
                'shopSecondFloor': 'Floor',
                'shopSideGate': 'Side',
                'shopSideSquare': 'Square',
                'southGate': 'Gate2'}        
    seq = ['circleRegion','shopSecondFloor', 'shopFrontGate','southGate','playground', 'movingView',\
                'park', 'innerShop',   'shopSideGate','shopSideSquare','test'    ]
    seq = ['circleRegion','shopSecondFloor', 'shopFrontGate','southGate','playground', 'movingView',\
                    'park', 'innerShop',   'shopSideGate','shopSideSquare','test'    ]
    seq = ['S2L1']
    #seq = ['MDMT26','MDMT31', 'MDMT34','MDMT48','MDMT52', 'MDMT55',\
             #       'MDMT56', 'MDMT57',   'MDMT59','MDMT61','MDMT62','MDMT68','MDMT71','MDMT73'    ]
    #seq = ['CAMPUS1','CAMPUS2', 'CAMPUS3'   ]
    seq = ['00003garden','00005garden',
           '00012night', '00018court1','00019court2',
           '00022gate','00023court4','00027square',
            '00028path','00029path','00043football',
            '00051football','00067football','00117wood',
            '00120park','00121park','00144road',
            '00156basketball','00161basketball',
            '00183canteen', '00225basketballdark',
            '00227librarydark','00229librarydark',
             '00231canteendark',  '00243institutedark'     ]
    view_out = ['Drone','View1','View2']
    #view_out = ['View1','View2','View3','View4']
    view_out = ['View1','View2']
    view_out = ['View1','View2','View3','View4','View5']
    view_out = ['View1','View2']#VISION
    for i in range(view_nums):
        for j in range(frames):
            index = j*view_nums
            instance = outputs[index+i]['instances'] 
            box = instance.pred_boxes.tensor
            track_ids = instance.track_ids
            if not os.path.exists(os.path.join(root,name)):
                os.mkdir(os.path.join(root, name))
            if not os.path.exists(os.path.join(result_root, seq[scene])):
                os.mkdir(os.path.join(result_root, seq[scene]))
            f = open(os.path.join(result_root, seq[scene], '{}.txt'.format(view_out[i])), 'a')
            #f = open(result_filename, 'w')
            for row in range(len(track_ids)):
                if i==1:
                    large = 1
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                    j+1, track_ids[row], box[row][0]*large, box[row][1]*large, box[row][2]*large, box[row][3]*large), file=f)         
                else :
                    large = 1
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                    j+1, track_ids[row], box[row][0]*large, box[row][1]*large, box[row][2]*large, box[row][3]*large), file=f)       
            #draw_rectangle(box,scene,i,j,track_ids,view_nums)                
        f.close() 
            
        
    

def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
       #     total_data_time += time.perf_counter() - start_data_time
  #          if idx == num_warmup:
    #            start_time = time.perf_counter()
        # #       total_data_time = 0
          #      total_compute_time = 0
            #    total_eval_time = 0

       #     start_compute_time = time.perf_counter()
            outputs ,view_nums= model(inputs)
            log_track(outputs,view_nums,idx)
            print(idx)
            del outputs
            #del inputs
            for i in range(len(inputs)):
                inputs[i]['image'] = None
            gc.collect()
        #    if torch.cuda.is_available():
        #        torch.cuda.synchronize()
    #if results is None:
    results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
