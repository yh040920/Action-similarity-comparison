import json
import time

import cv2
import numpy
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from calculate_similarity import calculate_similarity, select_piece_from_similarity
from topdown_demo_with_mmdet import main


class VideoThread(QThread):
    changePixmap = pyqtSignal(numpy.ndarray)
    emit_to_teacher = pyqtSignal(list)

    def __init__(self, parser):
        super(VideoThread, self).__init__()
        self._isStopped = False
        self.parser = parser

    def run(self):
        main(self.parser, input_dir='webcam', compare=True, changePixmap=self.changePixmap,
             emit_to_teacher=self.emit_to_teacher, self=self)

    def stop(self):
        self._isStopped = True
        self.exit()


class TeacherThread(QThread):
    update_label1 = pyqtSignal(numpy.ndarray)
    stop_contrast = pyqtSignal(bool)
    emit_all_scores = pyqtSignal(numpy.ndarray)

    def __init__(self, file_name, text_edit, check_group):
        super(TeacherThread, self).__init__()
        self.last_time = None
        self.points_teacher_all = []
        self.checkID = check_group.checkedId()
        self.points_teacher = None
        self.cap = cv2.VideoCapture(file_name)
        self.text_edit = text_edit
        self.is_ok = False
        self.index = 0
        self.all_student_points = []
        teacher_json_path = file_name.split('.')[0] + '.json'
        with open(teacher_json_path) as f:
            self.teacher_data = json.load(f)["instance_info"]

    def run(self):
        self.update_teacher()

    def update_teacher(self):

        ret, frame = self.cap.read()

        if not ret:
            self.stop_contrast.emit(True)
            if self.checkID == 2:
                similarity = calculate_similarity(np.array(self.points_teacher_all), np.array(self.all_student_points))
                piece_info = select_piece_from_similarity(similarity)
                all_scores = piece_info["similarity"]
                self.emit_all_scores.emit(all_scores)

            self.exit()
            return 0

        frame_data = self.teacher_data[self.index]['instances'][0]
        bbox = frame_data['bbox']
        x1, y1, x2, y2 = np.array(bbox).astype(int)[0]
        cropped_frame = frame[y1:y2, x1:x2]
        self.update_label1.emit(cropped_frame)
        teacher_keypoints = np.array(frame_data['keypoints'])
        teacher_keypoints_scores = np.array(frame_data['keypoint_scores'])
        if self.checkID == 1:
            self.points_teacher = np.concatenate((teacher_keypoints, teacher_keypoints_scores[..., None]), axis=-1). \
                reshape((1, 17, 3))
        else:
            self.points_teacher_all.append(np.concatenate((teacher_keypoints, teacher_keypoints_scores[..., None])
                                                          , axis=-1))

        # else:
        #     for i in self.teacher_data:
        #         frame_data = i['instances'][0]
        #         teacher_keypoints = np.array(frame_data['keypoints'])
        #         teacher_keypoints_scores = np.array(frame_data['keypoint_scores'])
        #         points_teacher = np.concatenate((teacher_keypoints, teacher_keypoints_scores[..., None]), axis=-1)

    def loop_list(self, list_):
        if self.checkID == 1:
            if self.is_ok:
                self.update_teacher()
                self.is_ok = False

            # if len(list_) != 1:
            #     self.text_edit.append("检测到多个目标")
            # else:
            keypoint_scores = np.array(list_[0]['keypoint_scores'])
            keypoints = np.array(list_[0]['keypoints'])
            points_student = np.concatenate((keypoints, keypoint_scores[..., None]), axis=-1).reshape((1, 17, 3))
            similarity = calculate_similarity(self.points_teacher, points_student)
            self.text_edit.append(f"score:{round(float(similarity), 2) * 100}")
            # if self.last_time:
            #     print(1 / (time.time()-self.last_time))
            # self.last_time = time.time()
            if similarity > 0.8:
                self.is_ok = True
                self.index += 1
        else:
            self.update_teacher()
            keypoint_scores = np.array(list_[0]['keypoint_scores'])
            keypoints = np.array(list_[0]['keypoints'])
            points_student = np.concatenate((keypoints, keypoint_scores[..., None]), axis=-1)
            self.all_student_points.append(points_student)
            self.index += 1
        print(self.index)

    def stop(self):
        self.exit()
