# main_function_handler.py
import psutil
import time
import os
import json
import numpy as np
import cv2
import zmq
import msgpack
from collections import defaultdict
from tarsyer_packages.shared_logger import SharedLogger
from tarsyer_packages.motion_detect import motion_detection


class ObjectDetector:
    def __init__(self, camera_vsq, sender, value_for_socket, centroid_tracker,
                 bbox_handler, heatmap_bbox_appender, heatmap_bbox_data_saver,
                 footfall_saver, debug_display, video_recorder, dwell_time_bbox_saver,
                 store_code, store, MIN_CONTOUR_AREA, NO_OBJECT_THRESH, NMS_THRESH, ZMQ_TIMEOUT=5000):
        self.camera_vsq = camera_vsq
        self.sender = sender
        self.value_for_socket = value_for_socket
        self.centroid_tracker = centroid_tracker

        self.bbox_handler = bbox_handler
        self.heatmap_bbox_appender = heatmap_bbox_appender
        self.heatmap_bbox_data_saver = heatmap_bbox_data_saver
        self.footfall_saver = footfall_saver
        self.debug_display = debug_display
        self.video_recorder = video_recorder
        self.dwell_time_bbox_saver = dwell_time_bbox_saver

        self.MIN_CONTOUR_AREA = MIN_CONTOUR_AREA
        self.NO_OBJECT_THRESH = NO_OBJECT_THRESH
        self.NMS_THRESH = NMS_THRESH
        self.ZMQ_TIMEOUT = ZMQ_TIMEOUT
        self.no_detection_output_counter = 0

        # Set ZMQ timeout
        self.sender.setsockopt(zmq.RCVTIMEO, self.ZMQ_TIMEOUT)

        self.logger = SharedLogger(store, "object_detector")

        # Initialize CPU tracking
        psutil.cpu_percent(interval=None)

    @staticmethod
    def get_next_interval_time(interval):
        current = time.time()
        return current + interval - (current % interval)

    @staticmethod
    def fps_calculation(counter, start_time):
        current_time = time.time()
        fps = round(counter / (current_time - start_time), 2)
        return fps, current_time

    @staticmethod
    def device_info():
        return (
            [round(num, 1) for num in os.getloadavg()],
            psutil.virtual_memory().percent,
            psutil.swap_memory().percent,
            int(psutil.sensors_temperatures()['cpu_thermal'][0].current),
            psutil.cpu_percent(interval=None)  # Non-blocking, returns usage since last call
        )

    @staticmethod
    def nms(dets_tup, scores, classes, thresh):
        if not dets_tup:
            return [], [], []

        dets = np.array(dets_tup)
        areas = (dets[:, 2] - dets[:, 0] + 1) * (dets[:, 3] - dets[:, 1] + 1)
        order = np.argsort(scores)[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(dets[i, 0], dets[order[1:], 0])
            yy1 = np.maximum(dets[i, 1], dets[order[1:], 1])
            xx2 = np.minimum(dets[i, 2], dets[order[1:], 2])
            yy2 = np.minimum(dets[i, 3], dets[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return [dets_tup[i] for i in keep], [scores[i] for i in keep], [classes[i] for i in keep]

    def handle_fps_calculation(self, current_time, next_fps_time, counter, start_time):
        if current_time >= next_fps_time:
            fps, start_time = self.fps_calculation(counter, start_time)
            self.logger.info(f"FPS:{fps}")

            counter = 0
            next_fps_time = self.get_next_interval_time(3)

            load_average, ram, swap_memory, cpu_temp, cpu_utilization = self.device_info()

            return fps, counter, next_fps_time, start_time, load_average, ram, swap_memory, cpu_temp, cpu_utilization
        return None

    def handle_json_saving(self, current_time, next_json_save, old_dict,
                           interval_based_detection_count, fps):
        if current_time >= next_json_save:
            old_dict = self.footfall_saver.save_json_data(
                old_dict, self.centroid_tracker, interval_based_detection_count, fps)

            heatmap_important_bboxes = self.heatmap_bbox_appender.get_bbox_list()
            self.heatmap_bbox_data_saver.save_bbox_data(heatmap_important_bboxes, fps)
            self.heatmap_bbox_appender.clear_bbox_list()

            next_json_save = self.get_next_interval_time(300)
            return old_dict, next_json_save
        return None

    def handle_motion_detection(self, motion_confirm, base_frame, resized_frame):
        returned_status, thresh, area = motion_detection(base_frame, resized_frame, self.MIN_CONTOUR_AREA)
        if returned_status:
            motion_confirm = True
        return motion_confirm, thresh

    def process_detections(self, motion_confirm, resized_frame, original_frame,
                           interval_based_detection_count, base_frame):
        """Process detections using single multi-class tracker and save crossing bboxes."""

        resized_frame_copy = resized_frame.copy()
        imp_bbox_lists, imp_class_lists, imp_scores_lists = [], [], []

        if motion_confirm:
            buffer = cv2.imencode('.jpg', resized_frame)[1].tobytes()
            try:
                self.sender.send_multipart([self.value_for_socket.encode('utf-8'), buffer])
                message = self.sender.recv()
                response = msgpack.unpackb(message)

                bbox_lists = response['bboxes']
                class_list = response['classes']
                scores = response['scores']

                interval_based_detection_count += 1

                if bbox_lists:
                    nms_bbox_list, nms_score_list, nms_class_list = self.nms(
                        bbox_lists, scores, class_list, self.NMS_THRESH)

                    class_id_to_name = {0: 'male', 1: 'female', 2: 'child', 3: 'staff'}

                    for bbox, score, class_id in zip(nms_bbox_list, nms_score_list, nms_class_list):
                        if self.bbox_handler.is_valid_bbox(bbox):
                            imp_bbox_lists.append(bbox)
                            imp_class_lists.append(class_id_to_name.get(class_id, 'unknown'))
                            imp_scores_lists.append(score)

                    self.heatmap_bbox_appender.add_bbox(imp_bbox_lists)

                    if not imp_bbox_lists:
                        self.no_detection_output_counter += 1
                        if self.no_detection_output_counter >= self.NO_OBJECT_THRESH:
                            self.no_detection_output_counter = 0
                            motion_confirm = False
                            base_frame = resized_frame_copy
                    else:
                        self.no_detection_output_counter = 0
                else:
                    self.no_detection_output_counter += 1
                    if self.no_detection_output_counter >= self.NO_OBJECT_THRESH:
                        self.no_detection_output_counter = 0
                        motion_confirm = False
                        base_frame = resized_frame_copy

            except zmq.error.Again:
                self.logger.error("ZMQ timeout occurred")
                motion_confirm = False
                base_frame = resized_frame_copy
        else:
            motion_confirm, _ = self.handle_motion_detection(motion_confirm, base_frame, resized_frame)

        # Update tracker and get crossing events
        objects, object_start_time, crossing_events = self.centroid_tracker.update(imp_bbox_lists, imp_class_lists)

        # Save bbox images for objects that crossed the line
        if crossing_events and original_frame is not None:
            for crossing_info in crossing_events:
                self.dwell_time_bbox_saver.save_crossing_bbox(original_frame, crossing_info)

        return (objects, object_start_time, motion_confirm,
                base_frame, interval_based_detection_count, imp_bbox_lists, imp_scores_lists)

    def handle_debug_display(self, debug_show, resized_frame, objects, imp_bbox_lists, imp_scores_lists,
                             motion_confirm, load_average, ram,
                             swap_memory, cpu_temp, cpu_utilization, fps, thresh):

        if debug_show:
            insights_img = self.debug_display.display_debug_info(
                resized_frame, objects, imp_bbox_lists, imp_scores_lists, motion_confirm, load_average,
                ram, swap_memory, cpu_temp, cpu_utilization, fps, self.centroid_tracker
            )
            cv2.imshow("insight_frame", insights_img)
            if thresh is not None:
                cv2.imshow('thresh', thresh)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return True, insights_img
        return False, None
