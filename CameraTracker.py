import csv
import os
import time

import cv2
import numpy as np
import json
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from Detection.Detection import ObjectDetection
from Visualization.visualization import draw_box, draw_caption, label_color

# from ReIdentification.re_identification import OverLappedView, NonOverLappedView
# # from Tracking.deep_sort import generate_detections_torch
# from Tracking.deep_sort import generate_detections_torch_resnet
# from Tracking.deep_sort import nn_matching
# from Tracking.deep_sort.Tracking_Deep_Sort import DeepSort
# from Tracking.deep_sort.detection import Detection as deep_sort_detection
# from Tracking.deep_sort.preprocessing import non_max_suppression


class MultiCameraTracker:

    def __init__(self, **parameters):
        self.parameters = parameters
        self._execution_path = os.getcwd()
        self._capture_cam1 = cv2.VideoCapture(parameters['input_video_path_cam1'])
        self._capture_cam2 = cv2.VideoCapture(parameters['input_video_path_cam2'])
        self.frame_width_cam1 = int(self._capture_cam1.get(3))
        self.frame_height_cam1 = int(self._capture_cam1.get(4))
        self.frame_width_cam2 = int(self._capture_cam2.get(3))
        self.frame_height_cam2 = int(self._capture_cam2.get(4))
        self._x_zoom_cam1 = self.parameters['zoom_bound_cam1'][0]
        self._y_zoom_cam1 = self.parameters['zoom_bound_cam1'][1]
        self._width_zoom_cam1 = self.parameters['zoom_bound_cam1'][2]
        self._height_zoom_cam1 = self.parameters['zoom_bound_cam1'][3]
        self._x_zoom_cam2 = self.parameters['zoom_bound_cam2'][0]
        self._y_zoom_cam2 = self.parameters['zoom_bound_cam2'][1]
        self._width_zoom_cam2 = self.parameters['zoom_bound_cam2'][2]
        self._height_zoom_cam2 = self.parameters['zoom_bound_cam2'][3]
        self._thick_cam1 = int((self.frame_height_cam1 + self.frame_width_cam1) // 300)
        self._thick_cam2 = int((self.frame_height_cam2 + self.frame_width_cam2) // 300)
        self._zoom_factor_cam1 = (self.frame_height_cam1 / self._height_zoom_cam1,
                                 self.frame_width_cam1 / self._width_zoom_cam1, 1)
        self._zoom_factor_cam2 = (self.frame_height_cam2 / self._height_zoom_cam2,
                                 self.frame_width_cam2 / self._width_zoom_cam2, 1)
        self._time_now = time.time()
        self.global_tracker_bboxes_id_cam1 = {}
        self.global_tracker_bboxes_id_cam2 = {}
        self._global_id_counter = 0
        self._frame_number = 0
        self._config_video()
        self._config_detection()
        self._config_tracking()
        self._config_multi_view()
        self._config_output_image()
        self._config_log()
        self._config_synchronism()

    def _config_video(self):
        input_name_cam1, _ = os.path.basename(self.parameters['input_video_path_cam1']).split('.')
        input_name_cam2, _ = os.path.basename(self.parameters['input_video_path_cam2']).split('.')
        output_name_video_cam1 = input_name_cam1 + '_' + self.parameters['detection_name'] + '_' + self.parameters[
            'tracker_name'] + '.avi'
        output_name_video_cam2 = input_name_cam2 + '_' + self.parameters['detection_name'] + '_' + self.parameters[
            'tracker_name'] + '.avi'
        output_name_join = 'join' + '_' + self.parameters['detection_name'] + '_' + self.parameters[
            'tracker_name'] + '.avi'
        output_video_dir = os.path.join(self._execution_path, self.parameters['output_video_dir'])
        if not os.path.exists(output_video_dir):
            os.makedirs(output_video_dir)
        output_video_path_cam1 = os.path.join(output_video_dir, output_name_video_cam1)
        output_video_path_cam2 = os.path.join(output_video_dir, output_name_video_cam2)
        output_video_path_join = os.path.join(output_video_dir, output_name_join)
        if self.parameters['output_video_resized']:
            output_video_width_cam1 = self.parameters['output_video_width']
            output_video_width_cam2 = self.parameters['output_video_width']
            output_video_height_cam1 = self.parameters['output_video_height']
            output_video_height_cam2 = self.parameters['output_video_height']
        else:
            output_video_width_cam1 = self.frame_width_cam1
            output_video_width_cam2 = self.frame_width_cam2
            output_video_height_cam1 = self.frame_height_cam1
            output_video_height_cam2 = self.frame_height_cam2
        output_video_width_join = output_video_width_cam1 + output_video_width_cam2
        output_video_height_join = max(output_video_height_cam1, output_video_height_cam2)

        self._output_video_cam1 = cv2.VideoWriter(output_video_path_cam1, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                              self.parameters['output_frame_per_second'],
                                              (output_video_width_cam1, output_video_height_cam1))
        self._output_video_cam2 = cv2.VideoWriter(output_video_path_cam2, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                              self.parameters['output_frame_per_second'],
                                              (output_video_width_cam2, output_video_height_cam2))
        self._output_video_join = cv2.VideoWriter(output_video_path_join, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                                  self.parameters['output_frame_per_second'],
                                                  (output_video_width_join, output_video_height_join))

    def _config_detection(self):
        detection_model_file = os.path.join(self._execution_path, self.parameters['detection_model_file'])
        if not os.path.isfile(detection_model_file):
            raise IOError('Detection model file not exist?')
        self._detector_cam1 = ObjectDetection(gpu_num=0)
        self._detector_cam2 = ObjectDetection(gpu_num=1)

        if self.parameters['detection_name'] == 'cascade_rcnn':
            self._detector_cam1.setModelTypeAsCascadeRCNN(self.parameters['config_file'])
            self._detector_cam2.setModelTypeAsCascadeRCNN(self.parameters['config_file'])
        else:
            raise Exception('invalid detection name.')

        self._detector_cam1.setModelPath(self.parameters['detection_model_file'])
        self._detector_cam2.setModelPath(self.parameters['detection_model_file'])
        self._detector_cam1.loadModel(detection_speed=self.parameters['detection_speed'])
        self._detector_cam2.loadModel(detection_speed=self.parameters['detection_speed'])
        self._custom_objects_cam1 = self._detector_cam1.CustomObjects(**self.parameters['custom_objects'])
        self._custom_objects_cam2 = self._detector_cam2.CustomObjects(**self.parameters['custom_objects'])
        self.numbers_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                                 6: 'train',
                                 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                                 12: 'parking meter',
                                 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                                 20: 'elephant',
                                 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                                 27: 'tie',
                                 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
                                 33: 'kite',
                                 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
                                 38: 'tennis racket',
                                 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                                 45: 'bowl',
                                 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                                 52: 'hot dog',
                                 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                                 59: 'bed',
                                 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                                 66: 'keyboard',
                                 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                                 72: 'refrigerator',
                                 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                                 78: 'hair dryer',
                                 79: 'toothbrush'}


    def _config_tracking(self):
        max_age = self.parameters['tracking_max_age']
        min_hits = self.parameters['tracking_min_hits']

        feature_model_file = os.path.join(self._execution_path, self.parameters['feature_model_file'])
        if not os.path.isfile(feature_model_file):
            raise IOError('Feature model file not exist?')
        max_iou_distance = self.parameters['tracking_max_iou_distance']
        matching_threshold = self.parameters['distance_metric_matching_threshold']
        buffer_size = self.parameters['distance_metric_buffer_size']
        self.metric_cam1 = nn_matching.NearestNeighborDistanceMetric("cosine", matching_threshold, buffer_size)
        self.metric_cam2 = nn_matching.NearestNeighborDistanceMetric("cosine", matching_threshold, buffer_size)
        self._tracker_cam1 = DeepSort(self.metric_cam1, max_iou_distance=max_iou_distance, max_age=max_age, n_init=min_hits)
        self._tracker_cam2 = DeepSort(self.metric_cam2, max_iou_distance=max_iou_distance, max_age=max_age, n_init=min_hits)
        self._encoder_cam1 = generate_detections_torch_resnet.create_box_encoder(feature_model_file)
        self._encoder_cam2 = generate_detections_torch_resnet.create_box_encoder(feature_model_file)
        self._nms_max_overlap = self.parameters['tracking_nms_max_overlap']


    def _config_multi_view(self):
        if self.parameters['multi_view_type'] == 'overlapped':
            homographi_matrix_file = os.path.join(self._execution_path, self.parameters['homographi_matrix_file'])
            if not os.path.isfile(homographi_matrix_file):
                raise IOError('Homographi matrix file not exist?')
            self._overLappedView = OverLappedView(homographi_matrix_file=homographi_matrix_file,
                                              map_kind=self.parameters['homographi_mapping_kind'],
                                              image_size_cam1=[self.frame_height_cam1, self.frame_width_cam1],
                                              image_size_cam2=[self.frame_height_cam2, self.frame_width_cam2],
                                              max_iou_distance=self.parameters['homographi_max_iou_distance'])

            self._image_separator_points_cam1 = self._overLappedView.image_separator_points_cam1
            self._image_separator_points_cam2 = self._overLappedView.image_separator_points_cam2

        elif self.parameters['multi_view_type'] == 'non_overlapped':
            self._nonOverLappedView = NonOverLappedView(image_size_cam1=[self.frame_height_cam1, self.frame_width_cam1],
                                          image_size_cam2=[self.frame_height_cam2, self.frame_width_cam2],
                                          metric_cam1=self.metric_cam1, metric_cam2=self.metric_cam2,
                                          min_age_exit=10, max_age_exit=500, min_age_enter_cam1=10,min_age_enter_cam2=15,
                                          max_age_enter=500, assignment_time_margin=70, distance_exit_cam1_to_enter_cam2=120,
                                          distance_exit_cam2_to_enter_cam1=0, max_feature_distance=0.6,
                                          exit_border_side_cam1='right', enter_border_side_cam1='right',
                                          exit_border_side_cam2='bottom', enter_border_side_cam2='bottom',
                                          exit_point_ratio_cam1=3 / 4, enter_point_ratio_cam1=5 / 8,
                                          exit_point_ratio_cam2=1 / 20, enter_point_ratio_cam2=1 / 5,
                                          direction_exit_cam1='east', direction_enter_cam1='west',
                                          direction_exit_cam2='west', direction_enter_cam2='east')

        else:
            raise Exception('Invalid multi view type.')



    def _config_output_image(self):
        self._output_image_dir = os.path.join(self._execution_path, self.parameters['output_image_dir'])
        if not os.path.exists(self._output_image_dir):
            os.makedirs(self._output_image_dir)

    def _config_log(self):
        output_log_dir = os.path.join(self._execution_path, self.parameters['output_log_dir'])
        if not os.path.exists(output_log_dir):
            os.makedirs(output_log_dir)
        self._log_info_file = open(os.path.join(output_log_dir, 'log.txt'), mode='w', encoding='utf-8')

        self._tracking_info_file = open(os.path.join(output_log_dir, 'tracking_info.csv'), mode='w', encoding='utf-8')
        self._tracking_info_writer = csv.writer(self._tracking_info_file, delimiter=',')
        self._tracking_info_writer.writerow(
            ['frame_number', 'camera_number', 'local_track_id', 'global_track_id', 'name', 'age',
             'velocity', 'direction', 'x_topLeft', 'y_topLeft', 'width', 'height'])
        self._tracking_info_file.flush()

    def _config_synchronism(self):
        self._tracker_bboxes_buffer_cam1 = []
        self._tracker_bboxes_buffer_cam2 = []
        self._tracker_bboxes_id_buffer_cam1 = []
        self._tracker_bboxes_id_buffer_cam2 = []


    def _Tracker(self, returned_frame, detected_objects_array, tracker, encoder):
        detections = []
        scores = []
        boxes_name = []
        for detection in detected_objects_array:
            current_box_points = detection.get('box_points')
            current_box_name = detection.get('name')
            percentage_probability = detection.get('percentage_probability')
            boxes_name.append(current_box_name)
            x1 = current_box_points[0]
            y1 = current_box_points[1]
            x2 = current_box_points[2]
            y2 = current_box_points[3]
            detections.append(np.array([x1, y1, x2 - x1, y2 - y1]).astype(np.float64))
            scores.append(percentage_probability / 100)

        detections = np.array(detections)
        scores = np.array(scores)

        features = encoder(returned_frame, detections.copy())
        detections = [deep_sort_detection(bbox, score, feature, name) for bbox, score, feature, name in
                      zip(detections, scores, features, boxes_name)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self._nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        trackers = tracker.update(detections)
        trackers_attributes = []
        for track in trackers:
            track_attributes = {}
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            track_attributes['id'] = int(track.track_id)
            track_attributes['bbox'] = [x1, y1, x2 - x1, y2 - y1]
            track_attributes['age'] = track.age
            track_attributes['name'] = track.name
            track_attributes['velocity_component'] = track.velocity_component
            track_attributes['velocity'] = track.velocity
            track_attributes['direction'] = track.direction
            track_attributes['hit_streak'] = track.hit_streak
            trackers_attributes.append(track_attributes)

        return trackers_attributes


    def _OverLapped_globalTracker(self, matches_bbox, unmatched_bbox_map, unmatched_bbox_candidate, homographi_mapping_kind,
                       trackers_attributes_cam1, trackers_attributes_cam2):

        tracker_bboxes_id_cam1 = [track_attributes['id'] for track_attributes in trackers_attributes_cam1]
        tracker_bboxes_id_cam2 = [track_attributes['id'] for track_attributes in trackers_attributes_cam2]
        tracker_ages_cam1 = [track_attributes['age'] for track_attributes in trackers_attributes_cam1]
        tracker_ages_cam2 = [track_attributes['age'] for track_attributes in trackers_attributes_cam2]

        if homographi_mapping_kind == '1to2':
            match_indices_cam1 = [idx for idx, _ in matches_bbox]
            match_indices_cam2 = [idx for _, idx in matches_bbox]
            common_area_unmatch_indices_cam1 = unmatched_bbox_map
            common_area_unmatch_indices_cam2 = unmatched_bbox_candidate
        elif homographi_mapping_kind == '2to1':
            match_indices_cam1 = [idx for _, idx in matches_bbox]
            match_indices_cam2 = [idx for idx, _ in matches_bbox]
            common_area_unmatch_indices_cam1 = unmatched_bbox_candidate
            common_area_unmatch_indices_cam2 = unmatched_bbox_map
        else:
            raise Exception('invalid mapping kind')

        common_area_indices_cam1 = match_indices_cam1 + common_area_unmatch_indices_cam1
        common_area_indices_cam2 = match_indices_cam2 + common_area_unmatch_indices_cam2
        non_common_arae_indices_cam1 = [idx for idx in tracker_bboxes_id_cam1 if idx not in common_area_indices_cam1]
        non_common_arae_indices_cam2 = [idx for idx in tracker_bboxes_id_cam2 if idx not in common_area_indices_cam2]

        for idx in non_common_arae_indices_cam1:
            if idx not in self.global_tracker_bboxes_id_cam1.keys():
                self._global_id_counter += 1
                self.global_tracker_bboxes_id_cam1[idx] = self._global_id_counter
        for idx in common_area_unmatch_indices_cam1:
            if idx not in self.global_tracker_bboxes_id_cam1.keys():
                self._global_id_counter += 1
                self.global_tracker_bboxes_id_cam1[idx] = self._global_id_counter

        for idx1, idx2 in zip(match_indices_cam1, match_indices_cam2):
            if idx1 not in self.global_tracker_bboxes_id_cam1.keys() and \
                    idx2 not in self.global_tracker_bboxes_id_cam2.keys():
                self._global_id_counter += 1
                self.global_tracker_bboxes_id_cam1[idx1] = self._global_id_counter
                self.global_tracker_bboxes_id_cam2[idx2] = self._global_id_counter
            elif idx1 in self.global_tracker_bboxes_id_cam1.keys() and \
                    idx2 not in self.global_tracker_bboxes_id_cam2.keys():
                self.global_tracker_bboxes_id_cam2[idx2] = self.global_tracker_bboxes_id_cam1[idx1]
            elif idx1 not in self.global_tracker_bboxes_id_cam1.keys() and \
                    idx2 in self.global_tracker_bboxes_id_cam2.keys():
                self.global_tracker_bboxes_id_cam1[idx1] = self.global_tracker_bboxes_id_cam2[idx2]
            else:
                if self.global_tracker_bboxes_id_cam1[idx1] != self.global_tracker_bboxes_id_cam2[idx2]:
                    age1 = tracker_ages_cam1[tracker_bboxes_id_cam1.index(idx1)]
                    age2 = tracker_ages_cam2[tracker_bboxes_id_cam2.index(idx2)]
                    if age1 >= age2:
                        self.global_tracker_bboxes_id_cam2[idx2] = self.global_tracker_bboxes_id_cam1[idx1]
                    else:
                        self.global_tracker_bboxes_id_cam1[idx1] = self.global_tracker_bboxes_id_cam2[idx2]

        for idx in common_area_unmatch_indices_cam2:
            if idx not in self.global_tracker_bboxes_id_cam2.keys():
                self._global_id_counter += 1
                self.global_tracker_bboxes_id_cam2[idx] = self._global_id_counter
        for idx in non_common_arae_indices_cam2:
            if idx not in self.global_tracker_bboxes_id_cam2.keys():
                self._global_id_counter += 1
                self.global_tracker_bboxes_id_cam2[idx] = self._global_id_counter



    def _NonOverLapped_globalTracker(self, assigned_id_cam1_to_cam2, assigned_id_cam2_to_cam1,
                                           entered_id_cam1, entered_id_cam2,
                                           trackers_attributes_cam1, trackers_attributes_cam2):

        tracker_bboxes_id_cam1 = [track_attributes['id'] for track_attributes in trackers_attributes_cam1]
        tracker_bboxes_id_cam2 = [track_attributes['id'] for track_attributes in trackers_attributes_cam2]

        exited_assigned_id_cam1 = []
        entered_assigned_id_cam1 = []
        exited_assigned_id_cam2 = []
        entered_assigned_id_cam2 = []
        for idx1, idx2 in assigned_id_cam1_to_cam2:
            exited_assigned_id_cam1.append(idx1)
            entered_assigned_id_cam2.append(idx2)
        for idx1, idx2 in assigned_id_cam2_to_cam1:
            exited_assigned_id_cam2.append(idx1)
            entered_assigned_id_cam1.append(idx2)

        entered_not_assigned_id_cam1 = [idx for idx in entered_id_cam1 if idx not in entered_assigned_id_cam1]
        entered_not_assigned_id_cam2 = [idx for idx in entered_id_cam2 if idx not in entered_assigned_id_cam2]

        custom_bboxes_id_cam1 = [idx for idx in tracker_bboxes_id_cam1 if idx not in entered_id_cam1]
        custom_bboxes_id_cam2 = [idx for idx in tracker_bboxes_id_cam2 if idx not in entered_id_cam2]

        for idx in custom_bboxes_id_cam1:
            if idx not in self.global_tracker_bboxes_id_cam1.keys():
                self._global_id_counter += 1
                self.global_tracker_bboxes_id_cam1[idx] = self._global_id_counter

        for idx in custom_bboxes_id_cam2:
            if idx not in self.global_tracker_bboxes_id_cam2.keys():
                self._global_id_counter += 1
                self.global_tracker_bboxes_id_cam2[idx] = self._global_id_counter

        for i, idx in enumerate(entered_assigned_id_cam1):
            ind = exited_assigned_id_cam2[i]
            self.global_tracker_bboxes_id_cam1[idx] = self.global_tracker_bboxes_id_cam2[ind]

        for i, idx in enumerate(entered_assigned_id_cam2):
            ind = exited_assigned_id_cam1[i]
            self.global_tracker_bboxes_id_cam2[idx] = self.global_tracker_bboxes_id_cam1[ind]

        # for idx in entered_not_assigned_id_cam1:
        #     self.global_tracker_bboxes_id_cam1[idx] = 0
        #
        # for idx in entered_not_assigned_id_cam2:
        #     self.global_tracker_bboxes_id_cam2[idx] = 0



    def _save_and_show_result(self, frame_cam1, frame_cam2,
                              detected_objects_array_cam1, detected_objects_array_cam2,
                              trackers_attributes_cam1, trackers_attributes_cam2):

        count_detection_cam1 = len(detected_objects_array_cam1)
        count_detection_cam2 = len(detected_objects_array_cam2)
        tracker_bboxes_id_cam1 = [track_attributes['id'] for track_attributes in trackers_attributes_cam1]
        tracker_bboxes_id_cam2 = [track_attributes['id'] for track_attributes in trackers_attributes_cam2]
        tracker_bboxes_cam1 = [track_attributes['bbox'] for track_attributes in trackers_attributes_cam1]
        tracker_bboxes_cam2 = [track_attributes['bbox'] for track_attributes in trackers_attributes_cam2]
        tracker_bboxes_cam1 = np.array(tracker_bboxes_cam1)
        tracker_bboxes_cam2 = np.array(tracker_bboxes_cam2)

        cv2.putText(frame_cam1, 'Frame: ' + str(self._frame_number), (10, 50), cv2.FONT_HERSHEY_DUPLEX,
                    1e-3 * self.frame_height_cam1, (0, 0, 0), self._thick_cam1 // 3)
        cv2.putText(frame_cam2, 'Frame: ' + str(self._frame_number), (10, 50), cv2.FONT_HERSHEY_DUPLEX,
                    1e-3 * self.frame_height_cam2, (0, 0, 0), self._thick_cam2 // 3)

        for detection in detected_objects_array_cam1:
            box_points = detection.get('box_points')
            score = detection.get('percentage_probability')
            label = detection.get('label')
            color = label_color(label)
            draw_box(frame_cam1, box_points, color=color, thickness=self.parameters['box_border_thickness'])
            if (self.parameters['display_object_name'] and self.parameters['display_percentage_probability']):

                caption = "{} {:.1f}".format(self.numbers_to_names[label], score)
                draw_caption(frame_cam1, box_points, caption)
            elif (self.parameters['display_object_name']):
                caption = "{} ".format(self.numbers_to_names[label])
                draw_caption(frame_cam1, box_points, caption)
            elif (self.parameters['display_percentage_probability']):
                caption = " {:.1f}".format(score)
                draw_caption(frame_cam1, box_points, caption)

        for detection in detected_objects_array_cam2:
            box_points = detection.get('box_points')
            score = detection.get('percentage_probability')
            label = detection.get('label')
            color = label_color(label)
            draw_box(frame_cam2, box_points, color=color, thickness=self.parameters['box_border_thickness'])
            if (self.parameters['display_object_name'] and self.parameters['display_percentage_probability']):

                caption = "{} {:.1f}".format(self.numbers_to_names[label], score)
                draw_caption(frame_cam2, box_points, caption)
            elif (self.parameters['display_object_name']):
                caption = "{} ".format(self.numbers_to_names[label])
                draw_caption(frame_cam2, box_points, caption)
            elif (self.parameters['display_percentage_probability']):
                caption = " {:.1f}".format(score)
                draw_caption(frame_cam2, box_points, caption)

        for i, idx in enumerate(tracker_bboxes_id_cam1):
            x = tracker_bboxes_cam1[i, 0]
            y = tracker_bboxes_cam1[i, 1]
            id = self.global_tracker_bboxes_id_cam1[idx] if self.parameters['show_global_id'] else idx
            cv2.putText(frame_cam1, str(id), (x, y),
                        0, 0.6e-3 * self.frame_height_cam1, (0, 255, 0), self._thick_cam1 // 6)

        for i, idx in enumerate(tracker_bboxes_id_cam2):
            x = tracker_bboxes_cam2[i, 0]
            y = tracker_bboxes_cam2[i, 1]
            id = self.global_tracker_bboxes_id_cam2[idx] if self.parameters['show_global_id'] else idx
            cv2.putText(frame_cam2, str(id), (x, y),
                        0, 0.6e-3 * self.frame_height_cam2, (0, 255, 0), self._thick_cam2 // 6)

        if self.parameters['show_separator_line']:
            cv2.line(frame_cam1,
                     (int(self._image_separator_points_cam1[0, 0]), int(self._image_separator_points_cam1[0, 1])),
                     (int(self._image_separator_points_cam1[1, 0]), int(self._image_separator_points_cam1[1, 1])),
                     (0, 255, 255), 2)
            cv2.line(frame_cam2,
                     (int(self._image_separator_points_cam2[0, 0]), int(self._image_separator_points_cam2[0, 1])),
                     (int(self._image_separator_points_cam2[1, 0]), int(self._image_separator_points_cam2[1, 1])),
                     (0, 255, 255), 2)

        if self.parameters['zoom_output']:
            frame_cam1, frame_cam2 = self._zoom(frame_cam1, frame_cam2)

        if self.parameters['output_video_resized']:
            frame_cam1 = cv2.resize(frame_cam1,
                                    (self.parameters['output_video_width'], self.parameters['output_video_height']))
            frame_cam2 = cv2.resize(frame_cam2,
                                    (self.parameters['output_video_width'], self.parameters['output_video_height']))

        if self.parameters['join_cemera']:
            detected_frame_join = np.hstack((frame_cam1, frame_cam2))

        if self.parameters['display']:
            if self.parameters['join_cemera']:
                cv2.imshow("Detection & Tracking", detected_frame_join)
            else:
                cv2.imshow("Detection & Tracking Camera1", frame_cam1)
                cv2.imshow("Detection & Tracking Camera2", frame_cam2)
            cv2.waitKey(1)

        if self.parameters['save_video']:
            if self.parameters['join_cemera']:
                self._output_video_join.write(detected_frame_join)
            else:
                self._output_video_cam1.write(frame_cam1)
                self._output_video_cam2.write(frame_cam2)

        if self.parameters['split_frame']:
            if self.parameters['join_cemera']:
                path = os.path.join(self._output_image_dir, "frame-{}.png".format(self._frame_number))
                cv2.imwrite(path, detected_frame_join)
            else:
                path_cam1 = os.path.join(self._output_image_dir, "image1_frame-{}.png".format(self._frame_number))
                path_cam2 = os.path.join(self._output_image_dir, "image2_frame-{}.png".format(self._frame_number))
                cv2.imwrite(path_cam1, frame_cam1)
                cv2.imwrite(path_cam2, frame_cam2)

        if self.parameters['log_info']:
            count_tracking_cam1 = len(tracker_bboxes_id_cam1)
            count_tracking_cam2 = len(tracker_bboxes_id_cam2)
            fps = 1 / (time.time() - self._time_now + 1e-16)
            self._time_now = time.time()
            line1 = "Camera1 : Processing Frame : {:d}, Fps :  {:.1f}, Count Detection : {:d}, Count Tracking : {:d}" \
                .format(self._frame_number, fps, count_detection_cam1, count_tracking_cam1)
            line2 = "Camera2 : Processing Frame : {:d}, Fps :  {:.1f}, Count Detection : {:d}, Count Tracking : {:d}" \
                .format(self._frame_number, fps, count_detection_cam2, count_tracking_cam2)
            self._log_info_file.writelines(line1 + '\n' + line2 + '\n')
            print(line1)
            print(line2)

        if self.parameters['write_tracking_info']:
            tracker_ages_cam1 = [track_attributes['age'] for track_attributes in trackers_attributes_cam1]
            tracker_ages_cam2 = [track_attributes['age'] for track_attributes in trackers_attributes_cam2]
            tracker_names_cam1 = [track_attributes['name'] for track_attributes in trackers_attributes_cam1]
            tracker_names_cam2 = [track_attributes['name'] for track_attributes in trackers_attributes_cam2]
            tracker_velocities_cam1 = [track_attributes['velocity'] for track_attributes in trackers_attributes_cam1]
            tracker_velocities_cam2 = [track_attributes['velocity'] for track_attributes in trackers_attributes_cam2]
            tracker_directions_cam1 = [track_attributes['direction'] for track_attributes in trackers_attributes_cam1]
            tracker_directions_cam2 = [track_attributes['direction'] for track_attributes in trackers_attributes_cam2]
            for i, idx in enumerate(tracker_bboxes_id_cam1):
                self._tracking_info_writer.writerow([self._frame_number, 1, idx, self.global_tracker_bboxes_id_cam1[idx],
                                                     tracker_names_cam1[i], tracker_ages_cam1[i],
                                                     round(tracker_velocities_cam1[i],1), tracker_directions_cam1[i],
                                                     tracker_bboxes_cam1[i, 0], tracker_bboxes_cam1[i, 1],
                                                     tracker_bboxes_cam1[i, 2], tracker_bboxes_cam1[i, 3]])
                self._tracking_info_file.flush()
            for i, idx in enumerate(tracker_bboxes_id_cam2):
                self._tracking_info_writer.writerow([self._frame_number, 2, idx, self.global_tracker_bboxes_id_cam2[idx],
                                                     tracker_names_cam2[i], tracker_ages_cam2[i],
                                                     round(tracker_velocities_cam2[i],1), tracker_directions_cam2[i],
                                                     tracker_bboxes_cam2[i, 0], tracker_bboxes_cam2[i, 1],
                                                     tracker_bboxes_cam2[i, 2], tracker_bboxes_cam2[i, 3]])
                self._tracking_info_file.flush()

    def _end_work(self):
        self._capture_cam1.release()
        self._capture_cam2.release()
        self._output_video_cam1.release()
        self._output_video_cam2.release()
        self._output_video_join.release()
        self._log_info_file.close()
        self._tracking_info_file.close()
        cv2.destroyAllWindows()
        print('End of work.')

    def _synchronism(self, trackers_attributes_cam1, trackers_attributes_cam2):

        tracker_bboxes_id_cam1 = [track_attributes['id'] for track_attributes in trackers_attributes_cam1]
        tracker_bboxes_id_cam2 = [track_attributes['id'] for track_attributes in trackers_attributes_cam2]
        tracker_bboxes_cam1 = [track_attributes['bbox'] for track_attributes in trackers_attributes_cam1]
        tracker_bboxes_cam2 = [track_attributes['bbox'] for track_attributes in trackers_attributes_cam2]
        tracker_bboxes_cam1 = np.array(tracker_bboxes_cam1)
        tracker_bboxes_cam2 = np.array(tracker_bboxes_cam2)

        self._tracker_bboxes_buffer_cam1.append(tracker_bboxes_cam1)
        self._tracker_bboxes_buffer_cam2.append(tracker_bboxes_cam2)
        self._tracker_bboxes_id_buffer_cam1.append(tracker_bboxes_id_cam1)
        self._tracker_bboxes_id_buffer_cam2.append(tracker_bboxes_id_cam2)
        buffer_size = self.parameters['synchronism_buffer_size']
        if self._frame_number == buffer_size:
            synchronism_matrix = np.zeros((buffer_size, buffer_size))
            for i in range(buffer_size):
                for j in range(buffer_size):
                    pass
                    # should change to correct input format
                    # _, _, _, matching_degree = self._commonObject(self._tracker_bboxes_buffer_cam1[i],
                    #                                               self._tracker_bboxes_buffer_cam2[j],
                    #                                               self._tracker_bboxes_id_buffer_cam1[i],
                    #                                               self._tracker_bboxes_id_buffer_cam2[j])
                    # synchronism_matrix[i, j] = matching_degree
            self._tracker_bboxes_buffer_cam1 = []
            self._tracker_bboxes_buffer_cam2 = []
            self._tracker_bboxes_id_buffer_cam1 = []
            self._tracker_bboxes_id_buffer_cam2 = []
            return synchronism_matrix
        else:
            return None


    def _zoom(self, frame_cam1, frame_cam2):

        cropped_cam1 = frame_cam1[self._y_zoom_cam1 : self._y_zoom_cam1 + self._height_zoom_cam1,
                                  self._x_zoom_cam1 : self._x_zoom_cam1 + self._width_zoom_cam1]

        cropped_cam2 = frame_cam2[self._y_zoom_cam2 : self._y_zoom_cam2 + self._height_zoom_cam2,
                                  self._x_zoom_cam2 : self._x_zoom_cam2 + self._width_zoom_cam2]

        frame_cam1 = cv2.resize(cropped_cam1, (self.frame_width_cam1, self.frame_height_cam1))
        frame_cam2 = cv2.resize(cropped_cam2, (self.frame_width_cam2, self.frame_height_cam2))

        return frame_cam1, frame_cam2


    # def _Detect(self, frame_cam, cam_number):
    #     if cam_number == 1:
    #         detected_frame_cam, detected_objects_array_cam = self._detector_cam1.detectCustomObjectsFromImage(
    #             custom_objects=self._custom_objects_cam1,
    #             input_image=frame_cam,
    #             input_type="array", output_type="array",
    #             minimum_percentage_probability=self.parameters['minimum_detection_probability'],
    #             display_percentage_probability=self.parameters['display_percentage_probability'],
    #             display_object_name=self.parameters['display_object_name'])
    #     else:
    #         detected_frame_cam, detected_objects_array_cam = self._detector_cam2.detectCustomObjectsFromImage(
    #             custom_objects=self._custom_objects_cam2,
    #             input_image=frame_cam,
    #             input_type="array", output_type="array",
    #             minimum_percentage_probability=self.parameters['minimum_detection_probability'],
    #             display_percentage_probability=self.parameters['display_percentage_probability'],
    #             display_object_name=self.parameters['display_object_name'])
    #
    #     return detected_frame_cam, detected_objects_array_cam





    def __call__(self, end_frame=None):
        while True:
            ret_cam1, frame_cam1 = self._capture_cam1.read()
            ret_cam2, frame_cam2 = self._capture_cam2.read()

            stop_condition = ret_cam1 is False or ret_cam2 is False or (cv2.waitKey(1) & 0xFF == ord('q')) or \
                             (end_frame is not None and self._frame_number >= end_frame)
            if stop_condition:
                self._end_work()
                break
            self._frame_number += 1

            check_frame_interval = self._frame_number % self.parameters['frame_detection_interval']

            if (self._frame_number == 1 or check_frame_interval == 0):
                frame_cam1 = cv2.cvtColor(frame_cam1, cv2.COLOR_BGR2RGB)
                frame_cam2 = cv2.cvtColor(frame_cam2, cv2.COLOR_BGR2RGB)

                # with ThreadPoolExecutor() as executor:
                #     future1 = executor.submit(self._Detect, frame_cam1, 1)
                #     future2 = executor.submit(self._Detect, frame_cam2, 2)
                # detected_frame_cam1, detected_objects_array_cam1 = future1.result()
                # detected_frame_cam2, detected_objects_array_cam2 = future2.result()

                detected_objects_array_cam1 = self._detector_cam1.detectCustomObjectsFromImage(
                    custom_objects=self._custom_objects_cam1,
                    input_image=frame_cam1,
                    minimum_percentage_probability=self.parameters['minimum_detection_probability'])


                detected_objects_array_cam2 = self._detector_cam2.detectCustomObjectsFromImage(
                    custom_objects=self._custom_objects_cam2,
                    input_image=frame_cam2,
                    minimum_percentage_probability=self.parameters['minimum_detection_probability'])


                trackers_attributes_cam1 = self._Tracker(frame_cam1, detected_objects_array_cam1,
                                                        self._tracker_cam1, self._encoder_cam1)
                trackers_attributes_cam2 = self._Tracker(frame_cam2, detected_objects_array_cam2,
                                                        self._tracker_cam2, self._encoder_cam2)

                if self.parameters['multi_view_type'] == 'overlapped':
                    matches_bbox, unmatched_bbox_map, unmatched_bbox_candidate, _ = \
                        self._overLappedView(trackers_attributes_cam1, trackers_attributes_cam2)

                    self._OverLapped_globalTracker(matches_bbox, unmatched_bbox_map, unmatched_bbox_candidate,
                                        self.parameters['homographi_mapping_kind'],
                                        trackers_attributes_cam1, trackers_attributes_cam2)

                elif self.parameters['multi_view_type'] == 'non_overlapped':
                    assigned_id_cam1_to_cam2, assigned_id_cam2_to_cam1, entered_id_cam1, entered_id_cam2 = \
                        self._nonOverLappedView(trackers_attributes_cam1, trackers_attributes_cam2,
                                                time_stamp=self._frame_number)

                    self._NonOverLapped_globalTracker(assigned_id_cam1_to_cam2, assigned_id_cam2_to_cam1,
                                                      entered_id_cam1, entered_id_cam2,
                                                      trackers_attributes_cam1, trackers_attributes_cam2)


                frame_cam1 = cv2.cvtColor(frame_cam1, cv2.COLOR_BGR2RGB)
                frame_cam2 = cv2.cvtColor(frame_cam2, cv2.COLOR_BGR2RGB)
                self._save_and_show_result(frame_cam1, frame_cam2,
                                           detected_objects_array_cam1, detected_objects_array_cam2,
                                           trackers_attributes_cam1, trackers_attributes_cam2)


                if self.parameters['do_synchronism']:
                    synchronism_matrix = self._synchronism(trackers_attributes_cam1, trackers_attributes_cam2)


class SingleCameraTracker:

    def __init__(self, **parameters):
        self.parameters = parameters
        self._execution_path = os.getcwd()
        self._capture_cam1 = cv2.VideoCapture(parameters['input_video_path_cam1'])
        self.frame_width_cam1 = int(self._capture_cam1.get(3))
        self.frame_height_cam1 = int(self._capture_cam1.get(4))
        self._x_zoom_cam1 = self.parameters['zoom_bound_cam1'][0]
        self._y_zoom_cam1 = self.parameters['zoom_bound_cam1'][1]
        self._width_zoom_cam1 = self.parameters['zoom_bound_cam1'][2]
        self._height_zoom_cam1 = self.parameters['zoom_bound_cam1'][3]
        self._thick_cam1 = int((self.frame_height_cam1 + self.frame_width_cam1) // 300)
        self._zoom_factor_cam1 = (self.frame_height_cam1 / self._height_zoom_cam1,
                                  self.frame_width_cam1 / self._width_zoom_cam1, 1)
        self._time_now = time.time()
        self.global_tracker_bboxes_id_cam1 = {}
        self._frame_number = 0
        self._config_video()
        self._config_detection()
        # self._config_tracking()
        self._config_output_image()
        self._config_log()


    def _config_video(self):
        input_name_cam1, _ = os.path.basename(self.parameters['input_video_path_cam1']).split('.')
        output_name_video_cam1 = input_name_cam1 + '_' + self.parameters['detection_name'] + '_' + self.parameters[
            'tracker_name'] + '.avi'
        output_video_dir = os.path.join(self._execution_path, self.parameters['output_video_dir'])
        if not os.path.exists(output_video_dir):
            os.makedirs(output_video_dir)
        output_video_path_cam1 = os.path.join(output_video_dir, output_name_video_cam1)
        if self.parameters['output_video_resized']:
            output_video_width_cam1 = self.parameters['output_video_width']
            output_video_height_cam1 = self.parameters['output_video_height']
        else:
            output_video_width_cam1 = self.frame_width_cam1
            output_video_height_cam1 = self.frame_height_cam1

        self._output_video_cam1 = cv2.VideoWriter(output_video_path_cam1, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                                  self.parameters['output_frame_per_second'],
                                                  (output_video_width_cam1, output_video_height_cam1))


    def _config_detection(self):
        detection_model_file = os.path.join(self._execution_path, self.parameters['detection_model_file'])
        if not os.path.isfile(detection_model_file):
            raise IOError('Detection model file not exist?')
        self._detector_cam1 = ObjectDetection(gpu_num=0)

        if self.parameters['detection_name'] == 'cascade_rcnn':
            self._detector_cam1.setModelTypeAsCascadeRCNN(self.parameters['config_file'])
        else:
            raise Exception('invalid detection name.')

        self._detector_cam1.setModelPath(self.parameters['detection_model_file'])
        self._detector_cam1.loadModel(detection_speed=self.parameters['detection_speed'])
        self._custom_objects_cam1 = self._detector_cam1.CustomObjects(**self.parameters['custom_objects'])
        self.numbers_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                                 6: 'train',
                                 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                                 12: 'parking meter',
                                 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                                 20: 'elephant',
                                 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                                 27: 'tie',
                                 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
                                 33: 'kite',
                                 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
                                 38: 'tennis racket',
                                 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                                 45: 'bowl',
                                 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                                 52: 'hot dog',
                                 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                                 59: 'bed',
                                 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                                 66: 'keyboard',
                                 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                                 72: 'refrigerator',
                                 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                                 78: 'hair dryer',
                                 79: 'toothbrush'}


    def _config_tracking(self):
        max_age = self.parameters['tracking_max_age']
        min_hits = self.parameters['tracking_min_hits']

        feature_model_file = os.path.join(self._execution_path, self.parameters['feature_model_file'])
        if not os.path.isfile(feature_model_file):
            raise IOError('Feature model file not exist?')
        max_iou_distance = self.parameters['tracking_max_iou_distance']
        matching_threshold = self.parameters['distance_metric_matching_threshold']
        buffer_size = self.parameters['distance_metric_buffer_size']
        self.metric_cam1 = nn_matching.NearestNeighborDistanceMetric("cosine", matching_threshold, buffer_size)
        self._tracker_cam1 = DeepSort(self.metric_cam1, max_iou_distance=max_iou_distance, max_age=max_age,
                                      n_init=min_hits)
        self._encoder_cam1 = generate_detections_torch_resnet.create_box_encoder(feature_model_file)
        self._nms_max_overlap = self.parameters['tracking_nms_max_overlap']


    def _config_output_image(self):
        self._output_image_dir = os.path.join(self._execution_path, self.parameters['output_image_dir'])
        if not os.path.exists(self._output_image_dir):
            os.makedirs(self._output_image_dir)

    def _config_log(self):
        output_log_dir = os.path.join(self._execution_path, self.parameters['output_log_dir'])
        if not os.path.exists(output_log_dir):
            os.makedirs(output_log_dir)
        self._log_info_file = open(os.path.join(output_log_dir, 'log.txt'), mode='w', encoding='utf-8')

        self._tracking_info_file = open(os.path.join(output_log_dir, 'tracking_info.csv'), mode='w', encoding='utf-8')
        self._tracking_info_writer = csv.writer(self._tracking_info_file, delimiter=',')
        self._tracking_info_writer.writerow(
            ['frame_number', 'track_id', 'name', 'age',
             'velocity', 'direction', 'x_topLeft', 'y_topLeft', 'width', 'height'])
        self._tracking_info_file.flush()



    def _Tracker(self, returned_frame, detected_objects_array, tracker, encoder):
        detections = []
        scores = []
        boxes_name = []
        for detection in detected_objects_array:
            current_box_points = detection.get('box_points')
            current_box_name = detection.get('name')
            percentage_probability = detection.get('percentage_probability')
            boxes_name.append(current_box_name)
            x1 = current_box_points[0]
            y1 = current_box_points[1]
            x2 = current_box_points[2]
            y2 = current_box_points[3]
            detections.append(np.array([x1, y1, x2 - x1, y2 - y1]).astype(np.float64))
            scores.append(percentage_probability / 100)

        detections = np.array(detections)
        scores = np.array(scores)

        features = encoder(returned_frame, detections.copy())
        detections = [deep_sort_detection(bbox, score, feature, name) for bbox, score, feature, name in
                      zip(detections, scores, features, boxes_name)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self._nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        trackers = tracker.update(detections)
        trackers_attributes = []
        for track in trackers:
            track_attributes = {}
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            track_attributes['id'] = int(track.track_id)
            track_attributes['bbox'] = [x1, y1, x2 - x1, y2 - y1]
            track_attributes['age'] = track.age
            track_attributes['name'] = track.name
            track_attributes['velocity_component'] = track.velocity_component
            track_attributes['velocity'] = track.velocity
            track_attributes['direction'] = track.direction
            trackers_attributes.append(track_attributes)

        return trackers_attributes


    def _save_and_show_result(self, frame_cam1, detected_objects_array_cam1, trackers_attributes_cam1):

        count_detection_cam1 = len(detected_objects_array_cam1)
        tracker_bboxes_id_cam1 = [track_attributes['id'] for track_attributes in trackers_attributes_cam1]
        tracker_bboxes_cam1 = [track_attributes['bbox'] for track_attributes in trackers_attributes_cam1]
        tracker_bboxes_cam1 = np.array(tracker_bboxes_cam1)

        cv2.putText(frame_cam1, 'Frame: ' + str(self._frame_number), (10, 50), cv2.FONT_HERSHEY_DUPLEX,
                    1e-3 * self.frame_height_cam1, (0, 0, 0), self._thick_cam1 // 3)


        for detection in detected_objects_array_cam1:
            box_points = detection.get('box_points')
            score = detection.get('percentage_probability')
            label = detection.get('label')
            color = label_color(label)
            draw_box(frame_cam1, box_points, color=color, thickness=self.parameters['box_border_thickness'])
            if (self.parameters['display_object_name'] and self.parameters['display_percentage_probability']):

                caption = "{} {:.1f}".format(self.numbers_to_names[label], score)
                draw_caption(frame_cam1, box_points, caption)
            elif (self.parameters['display_object_name']):
                caption = "{} ".format(self.numbers_to_names[label])
                draw_caption(frame_cam1, box_points, caption)
            elif (self.parameters['display_percentage_probability']):
                caption = " {:.1f}".format(score)
                draw_caption(frame_cam1, box_points, caption)


        for i, idx in enumerate(tracker_bboxes_id_cam1):
            x = tracker_bboxes_cam1[i, 0]
            y = tracker_bboxes_cam1[i, 1]
            cv2.putText(frame_cam1, str(idx), (x, y),
                        0, 0.6e-3 * self.frame_height_cam1, (0, 255, 0), self._thick_cam1 // 6)

        if self.parameters['zoom_output']:
            frame_cam1 = self._zoom(frame_cam1)

        if self.parameters['output_video_resized']:
            frame_cam1 = cv2.resize(frame_cam1,
                                             (self.parameters['output_video_width'],
                                              self.parameters['output_video_height']))


        if self.parameters['display']:
            cv2.imshow("Detection & Tracking Camera1", frame_cam1)
            cv2.waitKey(1)

        if self.parameters['save_video']:
            self._output_video_cam1.write(frame_cam1)


        if self.parameters['split_frame']:
            path_cam1 = os.path.join(self._output_image_dir, "image1_frame-{}.png".format(self._frame_number))
            cv2.imwrite(path_cam1, frame_cam1)


        if self.parameters['log_info']:
            count_tracking_cam1 = len(tracker_bboxes_id_cam1)
            fps = 1 / (time.time() - self._time_now + 1e-16)
            self._time_now = time.time()
            line1 = "Camera1 : Processing Frame : {:d}, Fps :  {:.1f}, Count Detection : {:d}, Count Tracking : {:d}" \
                .format(self._frame_number, fps, count_detection_cam1, count_tracking_cam1)
            self._log_info_file.writelines(line1 + '\n' )
            print(line1)


        if self.parameters['write_tracking_info']:
            tracker_ages_cam1 = [track_attributes['age'] for track_attributes in trackers_attributes_cam1]
            tracker_names_cam1 = [track_attributes['name'] for track_attributes in trackers_attributes_cam1]
            tracker_velocities_cam1 = [track_attributes['velocity'] for track_attributes in trackers_attributes_cam1]
            tracker_directions_cam1 = [track_attributes['direction'] for track_attributes in trackers_attributes_cam1]

            for i, idx in enumerate(tracker_bboxes_id_cam1):
                self._tracking_info_writer.writerow(
                    [self._frame_number, idx,
                     tracker_names_cam1[i], tracker_ages_cam1[i],
                     round(tracker_velocities_cam1[i], 1), tracker_directions_cam1[i],
                     tracker_bboxes_cam1[i, 0], tracker_bboxes_cam1[i, 1],
                     tracker_bboxes_cam1[i, 2], tracker_bboxes_cam1[i, 3]])
                self._tracking_info_file.flush()


    def _end_work(self):
        self._capture_cam1.release()
        self._output_video_cam1.release()
        self._log_info_file.close()
        self._tracking_info_file.close()
        cv2.destroyAllWindows()
        print('End of work.')


    def _zoom(self, frame_cam1):
        cropped_cam1 = frame_cam1[self._y_zoom_cam1: self._y_zoom_cam1 + self._height_zoom_cam1,
                       self._x_zoom_cam1: self._x_zoom_cam1 + self._width_zoom_cam1]

        frame_cam1 = cv2.resize(cropped_cam1, (self.frame_width_cam1, self.frame_height_cam1))
        return frame_cam1


    def _detect_from_image(self):
        input_image_dir = os.path.join(self._execution_path, self.parameters['input_image_path'])
        if not os.path.exists(input_image_dir):
            os.makedirs(input_image_dir)
        output_image_detection_dir = os.path.join(self._execution_path, self.parameters['output_image_detection_dir'])
        if not os.path.exists(output_image_detection_dir):
            os.makedirs(output_image_detection_dir)
        for path, subdirs, files in os.walk(input_image_dir):
            for file_name in files:
                image_file_path = os.path.join(path, file_name)
                frame = cv2.imread(image_file_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_copy = frame.copy()
                time_start = time.time()
                detected_objects_array = self._detector_cam1.detectCustomObjectsFromImage(
                    custom_objects=self._custom_objects_cam1,
                    input_image=frame,
                    minimum_percentage_probability=self.parameters['minimum_detection_probability'])

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = []
                info_list = []
                for i, detection in enumerate(detected_objects_array):
                    box_points = detection.get('box_points')
                    score = detection.get('percentage_probability')
                    label = detection.get('label')
                    color = label_color(label)
                    x1 = box_points[0]
                    y1 = box_points[1]
                    x2 = box_points[2]
                    y2 = box_points[3]
                    detections.append(np.array([x1, y1, x2 - x1, y2 - y1]).astype(np.float64))

                    info_list.append({'Type':self.numbers_to_names[label],
                                      'Detection probability': round(score,1),
                                      'Bounding box': box_points})

                    draw_box(frame, box_points, color=color, thickness=self.parameters['box_border_thickness'])
                    if (self.parameters['display_object_name'] and self.parameters['display_percentage_probability']):

                        caption = "{} {:.1f}".format(self.numbers_to_names[label], score)
                        draw_caption(frame, box_points, caption)
                    elif (self.parameters['display_object_name']):
                        caption = "{} ".format(self.numbers_to_names[label])
                        draw_caption(frame, box_points, caption)
                    elif (self.parameters['display_percentage_probability']):
                        caption = " {:.1f}".format(score)
                        draw_caption(frame, box_points, caption)

                    cv2.putText(frame, str(i), (x1, y1),
                            0, 0.3e-3 * self.frame_height_cam1, (0, 255, 0), self._thick_cam1 // 12)

                json_file_path = os.path.join(output_image_detection_dir, file_name.split('.')[0] + '.json')
                with open(json_file_path, mode='w', encoding='utf-8') as json_outfile:
                    json.dump(info_list, json_outfile, indent=4, ensure_ascii=False)

                detections = np.array(detections)
                # features = self._encoder_cam1(frame_copy, detections.copy())
                # cost_matrix = np.zeros((len(features), len(features)))
                # for i in range(len(features)):
                #     for j in range(len(features)):
                #         a = features[i,:].reshape((1,-1))
                #         b  = features[j,:].reshape((1,-1))
                #         cost_matrix[i,j] = nn_matching._cosine_distance(a, b, False)


                time_end = time.time()
                count_detection = len(detected_objects_array)
                if self.parameters['zoom_output']:
                    frame = self._zoom(frame)
                if self.parameters['output_image_resized']:
                    frame = cv2.resize(frame,
                                                (self.parameters['output_image_width'],
                                                 self.parameters['output_image_height']))


                if self.parameters['display']:
                    cv2.imshow("Detected Image", frame)
                    cv2.waitKey(1)

                if self.parameters['save_image']:
                    saved_file_path = os.path.join(output_image_detection_dir, file_name)
                    cv2.imwrite(saved_file_path, frame)

                if self.parameters['log_info']:
                    detection_time = time_end - time_start
                    self._time_now = time.time()
                    line = "Processed Image : {}, Detection Time :  {:.1f}, Count Detection : {:d}" \
                        .format(file_name, detection_time, count_detection)
                    self._log_info_file.writelines(line + '\n')
                    print(line)



    def __call__(self, end_frame=None):

        if self.parameters['detect_from_images']:
            self._detect_from_image()
            return

        while True:
            ret_cam1, frame_cam1 = self._capture_cam1.read()

            stop_condition = ret_cam1 is False or (cv2.waitKey(1) & 0xFF == ord('q')) or \
                             (end_frame is not None and self._frame_number >= end_frame)
            if stop_condition:
                self._end_work()
                break
            self._frame_number += 1

            check_frame_interval = self._frame_number % self.parameters['frame_detection_interval']
            if (self._frame_number == 1 or check_frame_interval == 0):

                frame_cam1 = cv2.cvtColor(frame_cam1, cv2.COLOR_BGR2RGB)
                detected_objects_array_cam1 = self._detector_cam1.detectCustomObjectsFromImage(
                    custom_objects=self._custom_objects_cam1, input_image=frame_cam1,
                    minimum_percentage_probability=self.parameters['minimum_detection_probability'])

                trackers_attributes_cam1 = self._Tracker(frame_cam1, detected_objects_array_cam1,
                                                         self._tracker_cam1, self._encoder_cam1)

                frame_cam1 = cv2.cvtColor(frame_cam1, cv2.COLOR_BGR2RGB)
                self._save_and_show_result(frame_cam1, detected_objects_array_cam1, trackers_attributes_cam1)




