# code_utils.py
import cv2
import numpy as np
import time
import json
from datetime import datetime, timedelta
import os
import shlex
import subprocess as sp
import base64
from tarsyer_packages.shared_logger import SharedLogger


# video saving
class VideoRecorder:
    def __init__(self, video_saving_dir, video_temp_saving_dir, video_input, store_code, store, saving_fps, store_cam):
        self.video_saving_dir = video_saving_dir
        self.video_temp_saving_dir = video_temp_saving_dir
        self.video_input = video_input
        self.store_code = store_code
        self.saving_fps = saving_fps
        self.store_cam = store_cam
        self.process = None
        self.start_time_record = time.time()
        os.makedirs(self.video_saving_dir, exist_ok=True)

        self.logger = SharedLogger(store, "video_recorder")
        self._start_new_process()

    def handle_video_recording(self, insights_img):
        if abs(time.time() - self.start_time_record) > 600:
            self._save_and_close_process()
            self._start_new_process()
        else:
            self.process.stdin.write(insights_img.tobytes())

    def close(self):
        if self.process:
            self._save_and_close_process()

    def _save_and_close_process(self):
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait()
            self.process.terminate()
            self.process = None
        os.system(f'mv /tmp/*mp4 {self.video_saving_dir}')

    def _start_new_process(self):
        saving_start_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        if 'video' in self.store_cam:
            output_filename = os.path.join(
                self.video_temp_saving_dir,
                f"STORE_{self.store_code}_{self.video_input.split('/')[-1].split('.')[0]}_{saving_start_date_time}_result.mp4"
            )
        else:
            output_filename = os.path.join(
                self.video_temp_saving_dir,
                f'STORE_{self.store_code}_{saving_start_date_time}_result.mp4'
            )

        self.process = sp.Popen(
            shlex.split(
                f'/usr/bin/ffmpeg -hide_banner -loglevel error -y -s 450x450 -pixel_format bgr24 -f rawvideo -r {self.saving_fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -crf 24 {output_filename}'
            ),
            stdin=sp.PIPE
        )
        self.start_time_record = time.time()
        self.logger.info(f"Started new video recording: {output_filename}")


# Gate ACTIVITY


class GateActivity:
    def __init__(self, store_code, device_serial_id, camera_name, open_time, close_time):
        self.store_code = store_code
        self.device_serial_id = device_serial_id
        self.camera_name = camera_name
        self.open_time = open_time
        self.close_time = close_time
        self.morning_saved = False
        self.evening_saved = False
        
    def _convert_time_to_minutes(self, time_str):
        """Convert time string like '09:00' to minutes from midnight"""
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    
    def _get_current_time_minutes(self):
        """Get current time in minutes from midnight"""
        now = datetime.now()
        return now.hour * 60 + now.minute
    
    def _frame_to_byte_string(self, frame):
        """Convert frame to base64 byte string"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _load_existing_json(self):
        """Load existing gate activity JSON from /tmp/gate_activity.json"""
        json_file_path = '/tmp/gate_activity.json'
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None
    
    def _save_gate_activity(self, activity_type, original_frame):
        """Save gate activity to JSON file in specified location"""
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H-%M-%S")
        
        # Create directory if it doesn't exist
        save_dir = "/tmp/gate_activity_json"
        os.makedirs(save_dir, exist_ok=True)
        
        # Create filename with specified format
        filename = f"{date}_{time}_{self.store_code}_{self.camera_name}_{activity_type}.json"
        file_path = os.path.join(save_dir, filename)
        
        image_byte_str = self._frame_to_byte_string(original_frame)
        
        gate_data = {
            "date_time": now.strftime("%Y-%m-%d_%H-%M-%S"),
            "store_code": self.store_code,
            "device_serial_id": self.device_serial_id,
            "camera_number": self.camera_name,
            "type": activity_type,
            "image_byte_str": image_byte_str,
            "footfall_count": 0,
            "gate_status": "close"
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(gate_data, f, indent=4)
            print(f"Gate activity saved: {activity_type} at {file_path}")
        except IOError as e:
            print(f"Error saving gate activity: {e}")
    
    def check_and_save_gate_activity(self, original_frame):
        """Check if it's time to save morning or evening gate activity"""
        current_time_minutes = self._get_current_time_minutes()
        open_time_minutes = self._convert_time_to_minutes(self.open_time)
        close_time_minutes = self._convert_time_to_minutes(self.close_time)
        
        # Check for morning activity (OPEN_TIME + 30 minutes)
        morning_check_time = open_time_minutes + 30
        if current_time_minutes == morning_check_time and not self.morning_saved:
            existing_data = self._load_existing_json()
            
            # Save if status is not "morning" or if JSON doesn't exist
            if not existing_data or existing_data.get("status") != "morning":
                self._save_gate_activity("morning", original_frame)
                self.morning_saved = True
        
        # Check for evening activity (CLOSE_TIME + 30 minutes)
        evening_check_time = close_time_minutes + 30
        if current_time_minutes == evening_check_time and not self.evening_saved:
            existing_data = self._load_existing_json()
            
            # Save if status is not "evening" or if JSON doesn't exist
            if not existing_data or existing_data.get("status") != "evening":
                self._save_gate_activity("evening", original_frame)
                self.evening_saved = True
    
    def reset_daily_flags(self):
        """Reset the saved flags at the start of a new day"""
        self.morning_saved = False
        self.evening_saved = False


# footfall json saving
class FootfallDataSaver:
    def __init__(self, store_code, store, CAMERA_NAME, DEVICE_SERIAL_ID, camera_no, json_storage_location='/tmp/footfall_storage/'):
        self.store_code = store_code
        self.camera_name = CAMERA_NAME
        self.camera_no = camera_no
        self.device_serial_id = DEVICE_SERIAL_ID
        self.json_storage_location = json_storage_location
        os.makedirs(self.json_storage_location, exist_ok=True)

        self.logger = SharedLogger(store, "footfall_saver")

    def saving_footfall_json(self, curr_dict):
        current_date_time = curr_dict["current_date_time"]
        positive_count_male = curr_dict["positive_count_male"]
        positive_count_female = curr_dict["positive_count_female"]
        positive_count_staff = curr_dict["positive_count_staff"]
        positive_count_child = curr_dict["positive_count_child"]
        negative_count_male = curr_dict["negative_count_male"]
        negative_count_female = curr_dict["negative_count_female"]
        negative_count_staff = curr_dict["negative_count_staff"]
        negative_count_child = curr_dict["negative_count_child"]
        fps = curr_dict["fps"]

        data_dict = {
            'date_time': current_date_time,
            'device_serial_id': self.device_serial_id,
            'store_code': self.store_code,
            'camera_no': self.camera_no,
            'count_male': positive_count_male,
            'count_female': positive_count_female,
            'count_staff': positive_count_staff,
            'count_child': positive_count_child,
            'opp_count_male': negative_count_male,
            'opp_count_female': negative_count_female,
            'opp_count_staff': negative_count_staff,
            'opp_count_child': negative_count_child,
            'fps': fps
        }

        filename = f'{self.json_storage_location}/{self.store_code}_{self.camera_name}_{current_date_time.replace(" ", "_").replace(":", "-")}.json'
        with open(filename, "w") as outfile:
            json.dump(data_dict, outfile, indent=4)

        self.logger.info(f"Saved footfall data: {filename}")

    def save_json_data(self, old_footfall_dict, centroid_tracker, interval_based_detection_count, fps):
        current_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        class_counts = centroid_tracker.get_class_counts()['classes']

        data_dict = {
            'current_date_time': current_date_time,
            'positive_count_male': class_counts['male']['pos'] - old_footfall_dict["positive_count_male"],
            'positive_count_female': class_counts['female']['pos'] - old_footfall_dict["positive_count_female"],
            'positive_count_staff': class_counts['staff']['pos'] - old_footfall_dict["positive_count_staff"],
            'positive_count_child': class_counts['child']['pos'] - old_footfall_dict["positive_count_child"],
            'negative_count_male': class_counts['male']['neg'] - old_footfall_dict["negative_count_male"],
            'negative_count_female': class_counts['female']['neg'] - old_footfall_dict["negative_count_female"],
            'negative_count_staff': class_counts['staff']['neg'] - old_footfall_dict["negative_count_staff"],
            'negative_count_child': class_counts['child']['neg'] - old_footfall_dict["negative_count_child"],
            'fps': fps
        }
        self.saving_footfall_json(data_dict)

        ori_data = {
            'current_date_time': current_date_time,
            'positive_count_male': class_counts['male']['pos'],
            'positive_count_female': class_counts['female']['pos'],
            'positive_count_staff': class_counts['staff']['pos'],
            'positive_count_child': class_counts['child']['pos'],
            'negative_count_male': class_counts['male']['neg'],
            'negative_count_female': class_counts['female']['neg'],
            'negative_count_staff': class_counts['staff']['neg'],
            'negative_count_child': class_counts['child']['neg'],
            'fps': fps
        }

        return ori_data


# bounding box checker
class BoundingBoxHandler:
    def __init__(self, bbox_x_limit, bbox_y_limit, store):
        self.bbox_x_limit = bbox_x_limit
        self.bbox_y_limit = bbox_y_limit
        self.logger = SharedLogger(store, "bbox_handler")

    def calculate_centroid(self, bbox):
        c_x = int((bbox[0] + bbox[2]) / 2)
        c_y = int((bbox[1] + bbox[3]) / 2)
        return c_x, c_y

    def is_valid_bbox(self, bbox):
        centroid_x, centroid_y = self.calculate_centroid(bbox)
        height = int(bbox[3] - bbox[1])
        width = int(bbox[2] - bbox[0])

        return ((self.bbox_x_limit[1] <= centroid_x < self.bbox_x_limit[0]) and
                (self.bbox_y_limit[1] <= centroid_y < self.bbox_y_limit[0]) and
                (self.bbox_y_limit[3] <= height < self.bbox_y_limit[2]) and
                (self.bbox_x_limit[3] <= width < self.bbox_x_limit[2]))


# heatmap bbox list appender
class HeatmapBBoxListAppender:
    def __init__(self, store):
        self.bbox_list = []
        self.logger = SharedLogger(store, "heatmap_appender")

    def add_bbox(self, bbox):
        self.bbox_list.append(bbox)

    def clear_bbox_list(self):
        self.bbox_list.clear()

    def get_bbox_list(self):
        return self.bbox_list


# heatmap bbox_saver
class HeatmapBBoxDataSaver:
    def __init__(self, store_code, store, camera_name, device_serial_id, cropping_coords, base_frame_size, heatmap_start,
                 json_storage_location='/tmp/heatmap_storage/'):
        self.store_code = store_code
        self.camera_name = camera_name
        self.device_serial_id = device_serial_id
        self.cropping_coords = cropping_coords
        self.base_frame_size = base_frame_size
        self.heatmap_start = heatmap_start
        self.json_storage_location = json_storage_location
        os.makedirs(self.json_storage_location, exist_ok=True)

        self.logger = SharedLogger(store, "heatmap_saver")

    def save_bbox_data(self, bbox_list, fps):
        if self.heatmap_start:
            current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            data_dict = {
                'date_time': current_date_time,
                'device_serial_id': self.device_serial_id,
                'store_code': self.store_code,
                'camera_no': self.camera_name,
                'image_size': self.base_frame_size,
                'cropping_coord': self.cropping_coords,
                'bbox_list': bbox_list,
                'fps': fps
            }

            filename = f'{self.json_storage_location}/{self.store_code}_{self.camera_name}_{current_date_time}.json'
            with open(filename, "w") as outfile:
                json.dump(data_dict, outfile, indent=4)

            self.logger.info(f"Saved heatmap data: {filename}")

        return True


# Dwell time bbox saver
class DwellTimeBBoxSaver:
    def __init__(self, store_code, store, camera_name, crop_coords, resize_shape, device_serial_id,
                 json_storage_location='/tmp/dwell_time_bbox_json/', footfall_config = '/home/celestial/retail_bootstrap_ui/ui/data/footfall_config.json'):
        self.store_code = store_code
        self.camera_name = camera_name
        self.crop_coords = crop_coords
        self.resize_shape = resize_shape
        self.json_storage_location = json_storage_location
        self.open_time, self.close_time = self.read_store_timings(footfall_config)
        self.gate_activity_saved = False
        self.device_serial_id = device_serial_id

        os.makedirs(self.json_storage_location, exist_ok=True)

        # Calculate normalization factors
        crop_width = crop_coords[2] - crop_coords[0] + 20
        crop_height = crop_coords[3] - crop_coords[1] + 20
        self.scale_x = crop_width / resize_shape[0]
        self.scale_y = crop_height / resize_shape[1]

        self.logger = SharedLogger(store, "dwell_time_bbox_saver")
        self.logger.info(f"Initialized DwellTimeBBoxSaver with scale_x={self.scale_x}, scale_y={self.scale_y}")


    def read_store_timings(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        store_timings = config.get("Store_timings", {})
        open_time = store_timings.get("open")
        close_time = store_timings.get("close")

        return open_time, close_time

    
    def save_crossing_bbox(self, original_frame, crossing_info):
        """Save bbox image when person crosses the line"""
        try:
            if crossing_info is None or crossing_info.get('bbox') is None:
                return

            bbox = crossing_info['bbox']
            direction = crossing_info['direction']

            # Denormalize bbox to crop coordinates
            x1 = int(bbox[0] * self.scale_x)
            y1 = int(bbox[1] * self.scale_y)
            x2 = int(bbox[2] * self.scale_x)
            y2 = int(bbox[3] * self.scale_y)

            # Adjust to original frame coordinates
            x1_orig = self.crop_coords[0] + x1
            y1_orig = self.crop_coords[1] + y1
            x2_orig = self.crop_coords[0] + x2
            y2_orig = self.crop_coords[1] + y2

            # Ensure coordinates are within frame bounds
            h, w = original_frame.shape[:2]
            x1_orig = max(0, min(x1_orig, w))
            y1_orig = max(0, min(y1_orig, h))
            x2_orig = max(0, min(x2_orig, w))
            y2_orig = max(0, min(y2_orig, h))

            # Crop bbox from original frame
            cropped_bbox = original_frame[y1_orig:y2_orig, x1_orig:x2_orig]

            if cropped_bbox.size == 0:
                self.logger.warning(f"Empty crop for objectID {crossing_info['objectID']}")
                return

            # Encode image to bytes
            _, buffer = cv2.imencode('.jpg', cropped_bbox)
            img_bytes = base64.b64encode(buffer).decode('utf-8')

            # Create json data
            current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            data_dict = {
                'date_time': current_date_time,
                'store_code': self.store_code,
                'camera_name': self.camera_name,
                'direction': direction,
                'object_class': crossing_info.get('class', 'unknown'),
                'objectID': crossing_info['objectID'],
                'image_bytes': img_bytes
            }

            # Save json file
            filename = f'{self.json_storage_location}/{current_date_time}_{self.camera_name}_{direction}.json'
            with open(filename, "w") as outfile:
                json.dump(data_dict, outfile, indent=4)

            self.logger.info(
                f"Saved crossing bbox for objectID {crossing_info['objectID']}, direction={direction}: {filename}")
            # Gate activity tracking - only for "pos" direction
            #if direction == "pos":
            #    self._save_gate_activity(original_frame, current_date_time)

        except Exception as e:
            self.logger.error(f"Error in save_crossing_bbox: {e}")


    def _save_gate_activity(self, original_frame, current_date_time):
        """Save gate activity data"""
        try:
            # Get current time
            current_time = datetime.now().time()
            
            # Parse open and close times
            open_hour, open_min = map(int, self.open_time.split(':'))
            close_hour, close_min = map(int, self.close_time.split(':'))
            
            open_time = datetime.now().replace(hour=open_hour, minute=open_min, second=0).time()
            close_time = datetime.now().replace(hour=close_hour, minute=close_min, second=0).time()
            
            

            # Calculate time ranges
            morning_start = (datetime.combine(datetime.today(), open_time) - timedelta(hours=2)).time()
            morning_end = (datetime.combine(datetime.today(), open_time) + timedelta(minutes=30)).time()
            evening_start = close_time
            evening_end = (datetime.combine(datetime.today(), close_time) + timedelta(minutes=30)).time()
            
            

            status = None
            if morning_start <= current_time <= morning_end:
                status = "morning"
            elif evening_start <= current_time <= evening_end:
                status = "evening"

            if status:
                # Save /tmp/gate_activity.json
                gate_activity_data = {
                    'date_time': current_date_time,
                    'status': status,
                    'camera_name': self.camera_name,
                    'saved': True
                }
                
                with open('/tmp/gate_activity.json', 'w') as f:
                    json.dump(gate_activity_data, f, indent=4)
                
                # Save detailed gate activity JSON only once
                if self.gate_activity_saved == False:
                    # Encode original frame to bytes
                    _, buffer = cv2.imencode('.jpg', original_frame)
                    img_bytes_original = base64.b64encode(buffer).decode('utf-8')
                    
                    detailed_data = {
                        "date_time": current_date_time,
                        "store_code": self.store_code,
                        "device_serial_id": self.device_serial_id,
                        "camera_number": self.camera_name,
                        "type": status,
                        "image_byte_str": img_bytes_original,
                        "footfall_count": 1,
                        "gate_status": "open"
                    }
                    
                    # Create directory if it doesn't exist
                    os.makedirs('/tmp/gate_activity_json', exist_ok=True)
                    
                    filename = f'/tmp/gate_activity_json/{current_date_time}_{self.store_code}_{self.camera_name}_{status}.json'
                    with open(filename, 'w') as f:
                        json.dump(detailed_data, f, indent=4)
                    
                    self.gate_activity_saved = True
                    self.logger.info(f"Saved gate activity: {filename}")
            else:
                self.gate_activity_saved = False
        except Exception as e:
            self.logger.error(f"Error in _save_gate_activity: {e}")


# debug display frames
class DebugDisplay:
    def __init__(self, line, skip_frame, logo_path, store):
        self.line = line
        self.skip_frame = skip_frame

        # Load and resize the logo to 100x100
        self.logo = cv2.imread(logo_path, -1)
        if self.logo is None:
            # Handle case where logo is not found, create a dummy one
            print(f"Warning: Logo not found at {logo_path}. Creating a dummy logo.")
            self.logo = np.zeros((100, 100, 4), dtype=np.uint8)
            self.logo[:, :, :] = (0, 255, 0, 255) # Green opaque
            cv2.putText(self.logo, "LOGO", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0, 255), 2)

        self.logo = cv2.resize(self.logo, (100, 100))
        self.alpha_s = self.logo[:, :, 3] / 255.0
        self.alpha_l = 1.0 - self.alpha_s

        # Logo position (top-right corner of the 450x450 insights_img area)
        # Assuming 320x320 main frame, the right section is from x=320 to x=450 (width 130)
        # Centering the 100x100 logo in this 130-pixel wide strip
        self.alpha_x1, self.alpha_x2 = 320 + (130 - 100) // 2, 320 + (130 - 100) // 2 + 100 # 335, 435
        self.alpha_y1, self.alpha_y2 = 10, 10 + 100 # 10, 110 (with 10px top padding)

        self.logger = SharedLogger(store, "debug_display")

    def display_debug_info(self, frame_for_detections, objects, imp_bbox_lists, imp_scores_lists, motion_confirm,
                           load_average, ram, swap_memory, cpu_temp, cpu_utilization, fps, centroid_tracker):
        """
        Display debug info using single multi-class tracker.
        The 'frame_for_detections' is expected to be 320x320 and already
        contain the drawn bounding boxes, centroids, lines, etc.
        """

        # Ensure the input frame is indeed 320x320
        if frame_for_detections.shape[0] != 320 or frame_for_detections.shape[1] != 320:
            self.logger.warning("Input 'frame_for_detections' is not 320x320. Resizing it now.")
            frame_for_detections = cv2.resize(frame_for_detections, (320, 320))

        motion_status_str, od_circle_color = ["OD", (0, 51, 255)] if motion_confirm else ["motion", (51, 255, 102)]

        # --- Draw the line ---
        cv2.line(frame_for_detections, self.line[0], self.line[1], (0, 0, 255), 2)

        # Draw bounding boxes
        for bbox, score in zip(imp_bbox_lists, imp_scores_lists):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame_for_detections, (x1, y2 - 20), (x2, y2 - 10), (100, 100, 0), -1)
            cv2.putText(frame_for_detections, f'{score:.2f}', (x1 + 5, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            cv2.rectangle(frame_for_detections, (x1, y1), (x2, y2), (250, 0, 0), 1)

        # Get current objects with class info
        objects_with_classes = centroid_tracker.get_current_objects_with_classes()

        # --- Draw bounding boxes and object info ---
        for obj_id, obj_data in objects_with_classes.items():
            centroid = obj_data['centroid']
            obj_class = obj_data['class']
            obj_time = int(time.time() - obj_data['start_time'])

            # Prepare text
            text = f"ID{obj_id} {obj_class[:1].upper()} T{obj_time}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Add padding around text
            padding = 2

            # Calculate rectangle coordinates based on text size
            rect_x1 = max(0, centroid[0] - padding)
            rect_y1 = max(0, centroid[1] - text_height - padding * 5)
            rect_x2 = min(319, centroid[0] + text_width + padding)
            rect_y2 = min(319, centroid[1] - padding)

            # Draw rectangle
            cv2.rectangle(frame_for_detections, (rect_x1, rect_y1),
                          (rect_x2, rect_y2), (100, 50, 100), -1)

            # Draw text (positioned at bottom-left of rectangle with padding)
            text_x = rect_x1 + padding
            text_y = rect_y2 - padding - baseline
            cv2.putText(frame_for_detections, text,
                        (text_x, text_y),
                        font, font_scale, (0, 0, 0), thickness)

            cv2.circle(frame_for_detections, tuple(centroid), 4, (0, 255, 0), -1)

            # Draw trajectory line
            start_point = obj_data['start_centroid']
            cv2.line(frame_for_detections, tuple(start_point), tuple(centroid), (0, 0, 255), 1)


        # Create insights image (450x450) with a blue background
        insights_img = np.full((450, 450, 3), (0, 138, 230), dtype=np.uint8)

        # Place the 320x320 frame with detections in the top-left corner
        # No resizing here, direct placement as requested for performance on RPi.
        insights_img[10:330, 5:325, :] = frame_for_detections

        # --- Device related settings and Logo (Top-Right Section: x=320 to 450, y=0 to 320) ---
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_stats = 0.4
        thickness_stats = 1
        stats_x_start = 325 + 5 # Start 5 pixels from the left edge of this section
        stats_y_offset = 120 # Start below the logo

        # Add logo to the top-right
        logo_y1, logo_y2 = self.alpha_y1, self.alpha_y2
        logo_x1, logo_x2 = self.alpha_x1, self.alpha_x2
        for c in range(0, 3):
            insights_img[logo_y1:logo_y2, logo_x1:logo_x2, c] = (
                    self.alpha_s * self.logo[:, :, c] + self.alpha_l * insights_img[logo_y1:logo_y2,
                                                                       logo_x1:logo_x2, c])

        # Display system stats
        cv2.putText(insights_img, f'FPS {round(fps, 1)} SF {self.skip_frame}',
                    (stats_x_start, stats_y_offset), font, font_scale_stats, text_color, thickness_stats)
        cv2.putText(insights_img, f'STATUS: {motion_status_str}',
                    (stats_x_start, stats_y_offset + 20), font, font_scale_stats, text_color, thickness_stats)
        cv2.circle(insights_img, (stats_x_start + 105, stats_y_offset + 15), 5, od_circle_color, -1) # Status indicator

        cv2.putText(insights_img, f'LA {load_average}',
                    (stats_x_start, stats_y_offset + 50), font, font_scale_stats, text_color, thickness_stats)
        cv2.putText(insights_img, f'RAM {ram} SM {swap_memory}',
                    (stats_x_start, stats_y_offset + 70), font, font_scale_stats, text_color, thickness_stats)
        cv2.putText(insights_img, f'CPU Temp {cpu_temp}',
                    (stats_x_start, stats_y_offset + 90), font, font_scale_stats, text_color, thickness_stats)
        cv2.putText(insights_img, f'CPU UTIL {cpu_utilization}',
                    (stats_x_start, stats_y_offset + 110), font, font_scale_stats, text_color, thickness_stats)

        # --- Tabular Class Counts (Bottom-Most Part: y=320 to 450) ---
        class_counts = centroid_tracker.get_class_counts()['classes']
        table_start_y = 350 # Y position for the table header, leaving some padding
        line_height = 20
        font_scale_table = 0.4
        thickness_table = 1

        # Column X positions for better alignment
        x_cls = 5
        x_pos = 60
        x_neg = 105
        x_ign = 150
        x_ext_pos = 205
        x_ext_neg = 265
        x_ext_ign = 325

        # Table Header
        cv2.putText(insights_img, "CLS", (x_cls, table_start_y), font, font_scale_table, text_color, thickness_table)
        cv2.putText(insights_img, "POS", (x_pos, table_start_y), font, font_scale_table, text_color, thickness_table)
        cv2.putText(insights_img, "NEG", (x_neg, table_start_y), font, font_scale_table, text_color, thickness_table)
        cv2.putText(insights_img, "IGN", (x_ign, table_start_y), font, font_scale_table, text_color, thickness_table)
        cv2.putText(insights_img, "EXT+", (x_ext_pos, table_start_y), font, font_scale_table, text_color, thickness_table)
        cv2.putText(insights_img, "EXT-", (x_ext_neg, table_start_y), font, font_scale_table, text_color, thickness_table)
        cv2.putText(insights_img, "EXTI", (x_ext_ign, table_start_y), font, font_scale_table, text_color, thickness_table)
        cv2.line(insights_img, (x_cls, table_start_y + 3), (445, table_start_y + 3), text_color, 1) # Separator line

        current_y = table_start_y + line_height

        # Display counts for each class
        # Ensure 'classes_list' is defined in your CentroidTracker
        for obj_class_name in centroid_tracker.classes_list:
            counts = class_counts.get(obj_class_name, {'pos': 0, 'neg': 0, 'ign': 0, 'ext_pos': 0, 'ext_neg': 0, 'ext_ign': 0})

            cv2.putText(insights_img, obj_class_name[:6].upper(), (x_cls, current_y), font, font_scale_table, text_color, thickness_table) # Truncate class name
            cv2.putText(insights_img, f'{counts["pos"]}', (x_pos, current_y), font, font_scale_table, text_color, thickness_table)
            cv2.putText(insights_img, f'{counts["neg"]}', (x_neg, current_y), font, font_scale_table, text_color, thickness_table)
            cv2.putText(insights_img, f'{counts["ign"]}', (x_ign, current_y), font, font_scale_table, text_color, thickness_table)
            cv2.putText(insights_img, f'{counts["ext_pos"]}', (x_ext_pos, current_y), font, font_scale_table, text_color, thickness_table)
            cv2.putText(insights_img, f'{counts["ext_neg"]}', (x_ext_neg, current_y), font, font_scale_table, text_color, thickness_table)
            cv2.putText(insights_img, f'{counts["ext_ign"]}', (x_ext_ign, current_y), font, font_scale_table, text_color, thickness_table)
            current_y += line_height

        return insights_img
