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


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


def rectangle(image, l, t, r, b, color, thick, label=None):
    """Draw a rectangle.

    Parameters
    ----------
    x : float | int
        Top left corner of the rectangle (x-axis).
    y : float | int
        Top let corner of the rectangle (y-axis).
    w : float | int
        Width of the rectangle.
    h : float | int
        Height of the rectangle.
    label : Optional[str]
        A text label that is placed at the top left corner of the
        rectangle.

    """
    pt1 = int(l), int(t)
    pt2 = int(r), int(b)
    cv2.rectangle(image, pt1, pt2, color, thick)
    if label is not None:
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, 1, 2)

        center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
              text_size[0][1]
        cv2.rectangle(image, pt1, pt2, color, -4)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                    thick, (255, 255, 255), thick)
    return image


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


def draw_rectangle(box, scene, view_num, frame, track_id, views):
    picture_root = './datasets/DIVO/images/test/'
    picture_root = './datasets/CAMPUS/images/test/'
    save = './datasets/DIVO/images/'
    save = './datasets/CAMPUS/images/'
    picture_root = './datasets/DIVO/images/test/'

    save = '/data1/zthkk/divo_gtr/GCT_V1/pic_results/'
    picture_root = './datasets/MDMT/images/test/'

    save = '/data1/zthkk/divo_gtr/GCT_V1/pic_results/'
    picture_root = './datasets/PETS09/images/test/'

    save = '/data1/zthkk/divo_gtr/GCT_V1/pic_results/'
    picture_root = './datasets/VisionTrack/images/test/'

    save = '/data1/zthkk/divo_gtr/GCT_V1/pic_results/'
    picture_root = './datasets/VisionTrack_T/images/test/'

    seqs = os.listdir(picture_root)
    seqs = sorted(seqs)
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
    num = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]  # DIVO
    num = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]  # MDMT
    num = [0]  # PETS09
    num = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36
        , 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68]  # VISIONTRACK
    seq = seqs[num[scene] + view_num]
    # img_path =  '{}/{}/img1/'.format(picture_root, seq)
    # images = os.listdir(img_path)
    # num_images = len([image for image in images if 'jpg' in image])
    file_name = '{}/img1V_cut/{}_{:06d}.jpg'.format(seq, seq, frame + 1)
    image_path = picture_root + file_name
    img = cv2.imread(image_path)
    name = 'PET18000112_720_25_objdetection0.525_multithred0.09_NMS0.65'
    name = 'TEST_VISIONT18000_2_640_60_objdetection0.525_multithred0.001_NMS0.65_MINLEN50'
    # name = 'test_code'
    save_name = os.path.join(save + name, seq)
    if not os.path.exists(os.path.join(save, name)):
        os.makedirs(os.path.join(save, name))
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    for row in range(len(track_id)):
        label = track_id[row]
        r, g, b = create_unique_color_uchar(label)
        if view_num == 1:
            img = rectangle(img, box[row][0], box[row][1], box[row][2], box[row][3], (r, g, b), 1,
                            str(int(track_id[row])))
        else:
            img = rectangle(img, box[row][0], box[row][1], box[row][2], box[row][3], (r, g, b), 1,
                            str(int(track_id[row])))
    pic_name = '/{}_{:06d}.jpg'.format(seq, frame + 1)
    cv2.imwrite(save_name + pic_name, img)


def log_track(outputs, view_nums, scene):
    frames = int(len(outputs) / view_nums)
    name = 'PET18000112_720_25_objdetection0.525_nodecay_multithred0.001_NMS0.65'
    # name = 'test_code'
    name = 'VISIONT18000_13_640_60_objdetection0.525_multithred0.001_NMS0.65_MINLEN50'
    root = './'
    result_root = root + name
    '''seqs_dict = {'circleRegion': 'Circle',
                 'innerShop': 'Shop',
                 'movingView': 'Moving',
                 'park': 'Park',
                 'playground': 'Ground',
                 'shopFrontGate': 'Gate1',
                 'shopSecondFloor': 'Floor',
                 'shopSideGate': 'Side',
                 'shopSideSquare': 'Square',
                 'southGate': 'Gate2'}
    seq = [     \
            'shopSideGate', 'shopSideSquare']
    seq = ['circleRegion', 'shopSecondFloor', 'shopFrontGate', 'southGate', 'playground', 'movingView', \
           'park', 'innerShop', 'shopSideGate', 'shopSideSquare']'''
    #seq = ['S2L1']
    # seq = ['MDMT26','MDMT31', 'MDMT34','MDMT48','MDMT52', 'MDMT55',\
    #       'MDMT56', 'MDMT57',   'MDMT59','MDMT61','MDMT62','MDMT68','MDMT71','MDMT73'    ]
    # seq = ['CAMPUS1','CAMPUS2', 'CAMPUS3'   ]
    seq = ['00001garden','00003garden', '00005garden',
           '00012night', '00018court1', '00019court2',
           '00022gate','00023court4', '00027square',
           '00028path','00029path', '00043football',
           '00051football', '00067football', '00117wood',
           '00120park', '00121park', '00144road','00146bridge',
           '00156basketball', '00161basketball',
           '00183canteen']
    '''seq = ['00002garden', '00004garden', '00006garden',
           '00014night', '00016court1','00017court1', '00020court3',
           '00021gate', '00024court4','00025square','00026square', '00027square',
           '00030path', '00031path', '00042football',
           '00050football', '00066football', '00116wood',
           '00118park', '00119park', '00145road', '00147bridge',
           '00157basketball', '00160basketball',
           '00182canteen']'''
    #seq = ['1', '2', '3','4', '5', '6']
    #seq = [ '7', '8', '9']
    #seq = ['10', '11', '12', '13']
    view_out = [ 'View1', 'View2']
    #seq = [ 'WILDTRACK']
    # view_out = ['View1','View2','View3','View4']
    #view_out = ['View1', 'View2']
    #view_out = ['View1', 'View2', 'View3', 'View4', 'View5']
    #view_out = ['View1', 'View2']  # VISION
    #view_out = ['1', '2', '3', '4']
    for i in range(view_nums):
        for j in range(frames):
            index = j * view_nums
            instance = outputs[index + i]['instances']
            box = instance.pred_boxes.tensor
            track_ids = instance.track_ids
            scores=instance.scores
            if not os.path.exists(os.path.join(root, name)):
                os.mkdir(os.path.join(root, name))
            if not os.path.exists(os.path.join(result_root, seq[scene])):
                os.mkdir(os.path.join(result_root, seq[scene]))
            f = open(os.path.join(result_root, seq[scene], '{}.txt'.format(view_out[i])), 'a')
            # f = open(result_filename, 'w')
            for row in range(len(track_ids)):
                if i == 1:
                    large = 1
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,-1' % (
                        j + 1, track_ids[row], box[row][0] * large, box[row][1] * large, box[row][2] * large,
                        box[row][3] * large,scores[row]), file=f)
                else:
                    large = 1
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,-1' % (
                        j + 1, track_ids[row], box[row][0] * large, box[row][1] * large, box[row][2] * large,
                        box[row][3] * large,scores[row]), file=f)
                    # draw_rectangle(box,scene,i,j,track_ids,view_nums)
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
            outputs, view_nums = model(inputs)
            log_track(outputs, view_nums, idx)
            print(idx)
            del outputs
            # del inputs
            for i in range(len(inputs)):
                inputs[i]['image'] = None
            gc.collect()
            #    if torch.cuda.is_available():
            #        torch.cuda.synchronize()
            '''total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

   # results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle'''
    # if results is None:
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