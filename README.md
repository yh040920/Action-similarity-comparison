# Introduction
This code is based on mmpose, using human posture key point recognition, and calculating the similarity of the two sets of identified key points for action comparison

# Procedures：

1、Follow the official MMPose installation method and install[MMPose](https://mmpose.readthedocs.io/en/latest/installation.html)。

2、Install dependencies in the created environment：`pip install -r requirements.txt`

3、Download[rtmdet checkpoint](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth)和[rtmpose checkpoint](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth)到checkpoints文件夹下。

4、run program.py。

