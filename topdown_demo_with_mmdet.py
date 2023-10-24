# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]

    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    if bboxes.size == 0:
        pass
    else:
        max_score_index = np.argmax(bboxes[:, -1])
        bboxes = bboxes[max_score_index, :].reshape(1, -1)

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main(parser_dict, **kwargs):
    input_dir = 'resource/sample_video.mp4'
    compare = False
    changePixmap = None
    emit_to_teacher = None
    self = None
    if 'input_dir' in kwargs:
        input_dir = kwargs['input_dir']
    if 'compare' in kwargs:
        compare = kwargs['compare']
    if 'changePixmap' in kwargs:
        changePixmap = kwargs["changePixmap"]
    if 'emit_to_teacher' in kwargs:
        emit_to_teacher = kwargs['emit_to_teacher']
    if 'self' in kwargs:
        self = kwargs['self']
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('--det_config', default=parser_dict['det_config'], help='Config '
                                                                                'file for'
                                                                                ' detection')
    parser.add_argument('--det_checkpoint',
                        default=parser_dict['det_checkpoint'], help='Checkpoint'
                                                                    ' file for '
                                                                    'detection')
    parser.add_argument('--pose_config', default=parser_dict['pose_config']
                        , help='Config file for pose')
    parser.add_argument('--pose_checkpoint', default=parser_dict['pose_checkpoint'], help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default=input_dir, help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='./result',
        help='root of the output img file. '
             'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=True,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default=parser_dict['device'], help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.8,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/' \
                              f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if input_type == 'image':

        # inference
        pred_instances = process_one_image(args, args.input, detector,
                                           pose_estimator, visualizer)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type in ['webcam', 'video']:

        if args.input == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(args.input)

        video_writer = None
        pred_instances_list = []
        frame_idx = 0

        while cap.isOpened():

            if self and self._isStopped:
                cap.release()
                break

            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break

            # topdown pose estimation
            pred_instances = process_one_image(args, frame, detector,
                                               pose_estimator, visualizer,
                                               0.001)
            if emit_to_teacher:
                emit_to_teacher.emit(split_instances(pred_instances))
            """print(split_instances(pred_instances))： [{'keypoints': [[681.9034423828125, 191.031982421875], 
            [691.5744018554688, 179.97943115234375], [669.4692993164062, 181.3610076904297], [705.3900756835938, 
            179.97943115234375], [650.1273193359375, 185.50570678710938], [720.5873413085938, 225.57119750976562], 
            [630.785400390625, 225.57119750976562], [744.073974609375, 265.6366882324219], [665.3245849609375, 
            254.58413696289062], [662.5614624023438, 262.87353515625], [716.442626953125, 243.53158569335938], 
            [716.442626953125, 377.5437316894531], [661.1798706054688, 380.3068542480469], [713.6795043945312, 
            510.1742858886719], [705.3900756835938, 501.8848876953125], [692.9559936523438, 633.1338500976562], 
            [641.8379516601562, 606.8840942382812]], 'keypoint_scores': [0.9089821577072144, 0.9379405975341797, 
            0.9409140944480896, 0.9039398431777954, 0.9188470840454102, 0.7589665651321411, 0.8105413913726807, 
            0.7849964499473572, 0.45757824182510376, 0.6411226987838745, 0.5632444620132446, 0.7933688163757324, 
            0.7846053838729858, 0.8681372404098511, 0.9251211881637573, 0.9094620943069458, 0.9097778797149658], 
            'bbox': ([601.8285522460938, 126.37457275390625, 759.2151489257812, 692.2650146484375],), 'bbox_score': 
            1.0}]"""
            if args.save_predictions and not compare:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)))

            frame_vis = visualizer.get_image()  # frame_vis是渲染过后的图片
            if changePixmap:
                changePixmap.emit(frame_vis)
            # output videos
            if output_file and not compare:

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        25,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

        cap.release()

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions and not compare:
        with open(args.pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')

    if output_file and not compare:
        input_type = input_type.replace('webcam', 'video')
        print_log(
            f'the output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    parser = {'det_config': './mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py',
              'det_checkpoint': './checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
              'pose_config': './checkpoints/rtmpose/coco/rtmpose-m_8xb256-420e_aic-coco-256x192.py',
              'pose_checkpoint': './checkpoints/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192'
                                 '-63eb25f7_20230126.pth',
              'device': 'cuda:0'}
    main(parser)
