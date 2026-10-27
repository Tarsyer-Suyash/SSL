# config.py
import argparse
import json
import os
import zmq
import time

from tarsyer_packages.video_stream_queue import VideoStreamQueue
from tarsyer_packages.centroid_tracker import CentroidTracker
from tarsyer_packages.code_utils import (
    VideoRecorder, FootfallDataSaver, BoundingBoxHandler,
    HeatmapBBoxListAppender, HeatmapBBoxDataSaver,
    DwellTimeBBoxSaver, DebugDisplay, GateActivity
)
from tarsyer_packages.shared_logger import SharedLogger






# Taking input through argparser
parser = argparse.ArgumentParser(
    description="Process video streams for footfall and heatmap generation"
)

parser.add_argument('--selected_store_cam', type=str, required=True)
parser.add_argument('--video_name', type=str, default=None)

parser.add_argument('--footfall_save_start', type=bool, default=True)
parser.add_argument('--heatmap_start', type=bool, default=False)

parser.add_argument('--debug_screen', type=bool, default=False)
parser.add_argument('--video_write', type=bool, default=False)


# Parse arguments
args = parser.parse_args()
STORE_CAM = args.selected_store_cam
video_name = args.video_name
print(f"Store Code :: {STORE_CAM}")
print(f"Video name :: {video_name}")

arg_heatmap_start = args.heatmap_start
arg_SEND_DATA = args.footfall_save_start
arg_DEBUG_SHOW = args.debug_screen
arg_VIDEO_WRITE = args.video_write

# Checking if arg contain video processing
if not "video" in STORE_CAM:
    STORE = STORE_CAM
else:
    STORE = STORE_CAM.split("-")[1]

# Loading the config file
with open(f'/home/celestial/retail_bootstrap_ui/ui/data/{STORE}.json', "r") as f:
    all_data_dict = json.load(f)
all_data_dict_key = list(all_data_dict.keys())

# Set video input
if not "video" in STORE_CAM:
    VIDEO_INPUT = all_data_dict["rtsp_link"]
    METHOD = "feed"
else:
    VIDEO_INPUT = video_name
    METHOD = "video"

TARGET_FPS = 6
MAX_CAMERA_FPS = 50

# Store details
STORE_CODE = all_data_dict["store_code"]
# Add "ChannelNo" prefix if not present
camera_name_str = str(all_data_dict["camera_name"]).zfill(2)
CAMERA_NAME = camera_name_str if 'ChannelNo' in camera_name_str else f"ChannelNo{camera_name_str}"
DEVICE_SERIAL_ID = os.popen('sudo cat /sys/firmware/devicetree/base/serial-number').read().replace('\x00', '')

# Store_timings
OPEN_TIME = all_data_dict['Store_timings']['open']
CLOSE_TIME = all_data_dict['Store_timings']['close']

# Centroid tracker parameters
CENTROID_TRACKER_MAXDISAPPEARED = all_data_dict["centroid_tracker_maxDisappeared"]
CENTROID_TRACKER_MAXDISTANCE = all_data_dict["centroid_tracker_maxDistance"]
CENTROID_TRACKER_MINDISTANCE = all_data_dict["centroid_tracker_minDistance"]

# Bounding box limits
bbox_x_limit = [all_data_dict["bbox_x_limit_upper"], all_data_dict["bbox_x_limit_lower"],
                all_data_dict["bbox_x_limit_max_width"], all_data_dict["bbox_x_limit_min_width"]]
bbox_y_limit = [all_data_dict["bbox_y_limit_upper"], all_data_dict["bbox_y_limit_lower"],
                all_data_dict["bbox_y_limit_max_height"], all_data_dict["bbox_y_limit_min_height"]]

# Model related parameters
CONFIDENCE_THRESH = all_data_dict["confidence_thresh"]
NMS_THRESH = all_data_dict["nms_thresh"]
centroid_point = all_data_dict["centroid"]

# ROI details
CROPING_COORD_PRI = all_data_dict["crop_coord"]
LINE = all_data_dict["line"]
LINE = [tuple(LINE[0]), tuple(LINE[1])]
entry_point = all_data_dict["entry_point"]

# FPS and skip frame
camera_fps = int(all_data_dict["camera_fps"])
SKIP_FRAME = max(int(round(camera_fps/6, 0)), 1)

# Model input parameters
HEIGHT, WIDTH = 320, 320
INPUT_SHAPE = [HEIGHT, WIDTH]

# Code parameters
DEBUG_SHOW = arg_DEBUG_SHOW
HEATMAP_START = arg_heatmap_start
SEND_DATA = arg_SEND_DATA

old_dict = {
    'current_date_time': "2024-10-03_12-12-12",
    'positive_count_male': 0,
    'positive_count_female': 0,
    'positive_count_child': 0,
    'positive_count_staff': 0,
    'negative_count_male': 0,
    'negative_count_female': 0,
    'negative_count_staff': 0,
    'negative_count_child': 0,
    'fps': 0
}

# Motion detector
MIN_CONTOUR_AREA = 1000
motion_confirm = False
motion_thresh_counter = 0
MOTION_CONFIRM_THRESH = 3
no_detection_output_counter = 0
NO_OBJECT_THRESH = (camera_fps / SKIP_FRAME) * 5

# Video write
VIDEO_WRITE = arg_VIDEO_WRITE

# Initialize SharedLogger for main config
config_logger = SharedLogger(STORE, "config")
config_logger.info(f"Starting initialization for store: {STORE}")

## Initializing all the classes
# Initialize video stream queue
camera_vsq = VideoStreamQueue(VIDEO_INPUT, STORE, CROPING_COORD_PRI, SKIP_FRAME, TARGET_FPS, METHOD, MAX_CAMERA_FPS)
camera_vsq.start()
time.sleep(2)

# Get initial frame info
initial_frame_dict = camera_vsq.read()
uncropped_base_frame_size = initial_frame_dict['uncropped_base_frame_size']
base_frame = initial_frame_dict['image']

class_names = ['male', 'female', 'child', 'staff']


#Initialize gate activity
gate_activity = GateActivity(STORE_CODE, DEVICE_SERIAL_ID, CAMERA_NAME, OPEN_TIME, CLOSE_TIME)

# Initialize centroid tracker
centroid_tracker = CentroidTracker(
    store=STORE,
    line=LINE,
    entry_point=entry_point,
    centroid_point=centroid_point,
    classes_list=class_names,
    counting_axis='x',
    maxDisappeared=CENTROID_TRACKER_MAXDISAPPEARED,
    maxDistance=CENTROID_TRACKER_MAXDISTANCE,
    minDistance=CENTROID_TRACKER_MINDISTANCE)

# Initialize ZMQ
context = zmq.Context()
sender = context.socket(zmq.DEALER)
value_for_socket = f"{STORE}"
sender.setsockopt(zmq.IDENTITY, value_for_socket.encode('utf-8'))
sender.connect("tcp://127.0.0.1:5555")

# Initialize heatmap appender
heatmap_bbox_appender = HeatmapBBoxListAppender(STORE)

# Initialize heatmap data saver
heatmap_bbox_data_saver = HeatmapBBoxDataSaver(
    STORE_CODE, STORE, STORE_CAM, DEVICE_SERIAL_ID, CROPING_COORD_PRI, uncropped_base_frame_size, HEATMAP_START)

# Initialize footfall json saver
footfall_saver = FootfallDataSaver(STORE_CODE, STORE, CAMERA_NAME, DEVICE_SERIAL_ID, int(all_data_dict["camera_name"]))

# Initialize DebugDisplay
logo_path = "Tarsyer_Logo_BrandName.png"
debug_display = DebugDisplay(LINE, SKIP_FRAME, logo_path, STORE)

# Initialize BoundingBoxHandler
bbox_handler = BoundingBoxHandler(bbox_x_limit, bbox_y_limit, STORE)

# Initialize DwellTimeBBoxSaver
dwell_time_bbox_saver = DwellTimeBBoxSaver(
    STORE_CODE, STORE, CAMERA_NAME, CROPING_COORD_PRI, (320, 320), DEVICE_SERIAL_ID)

# Initialize video recording
if VIDEO_WRITE:
    saving_fps_int = int(round(camera_fps/SKIP_FRAME, 0))
    print(f"saving fps :: {saving_fps_int}")
    video_recorder = VideoRecorder(
        video_saving_dir='/tmp/videos',
        video_temp_saving_dir='/tmp',
        video_input=VIDEO_INPUT,
        store_code=STORE_CODE,
        store=STORE,
        saving_fps=saving_fps_int,
        store_cam=STORE_CAM
    )
else:
    video_recorder = None

# Initialize motion detector
motion_confirm = False
no_detection_output_counter = 0

# Initialize variables
frame_counter = 0
interval_based_detection_count = 0

# Initialize fps parameters
fps = 0
counter = 1
start_time = time.time()
time.sleep(2)

from tarsyer_packages.main_function_handler import *

# Create ObjectDetector instance
detector = ObjectDetector(
    camera_vsq, sender, value_for_socket,
    centroid_tracker,
    bbox_handler, heatmap_bbox_appender, heatmap_bbox_data_saver, footfall_saver,
    debug_display, video_recorder, dwell_time_bbox_saver,
    STORE_CODE, STORE, MIN_CONTOUR_AREA, NO_OBJECT_THRESH, NMS_THRESH
)

# Initialize timers to next exact intervals
next_fps_time = detector.get_next_interval_time(3)
next_json_save = detector.get_next_interval_time(300)
load_average, ram, swap_memory, cpu_temp, cpu_utilization = detector.device_info()

config_logger.info("Initialization complete")
