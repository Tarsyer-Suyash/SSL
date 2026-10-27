# video_stream_queue.py
import cv2
from threading import Thread
from queue import Queue
import time
from tarsyer_packages.shared_logger import SharedLogger


class VideoStreamQueue:
    def __init__(self, video_input, store, crop_coord_pri, skip_frame=5, target_fps=6, method='feed',
                 max_camera_fps=50):

        self.video_input = video_input
        self.setup_logging(store)
        self.crop_coord_pri = crop_coord_pri
        self.SKIP_FRAME = skip_frame
        self.skip_frame = skip_frame
        self.target_fps = target_fps
        self.target_cycle_time = 1 / target_fps
        self.method = method
        self.max_camera_fps = max_camera_fps
        self.last_log_time = time.perf_counter()
        self.frames_read = 0

        stream_status = "GPU"
        self.stream = cv2.VideoCapture(self.video_input, cv2.CAP_FFMPEG,
                                       [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_DRM])
        if not self.stream.isOpened():
            stream_status = "CPU"
            self.stream = cv2.VideoCapture(self.video_input, cv2.CAP_FFMPEG)

        self.vsq_logger.info(f"Streaming using {stream_status}")

        self.stopped = False
        self.queue = Queue(maxsize=1)
        self.thread = Thread(target=self.update, daemon=True)

    def setup_logging(self, store):
        """Set up logging configuration using SharedLogger"""
        self.vsq_logger = SharedLogger(store, "video_stream_queue")
        self.vsq_logger.info(f"Video Input :: {self.video_input}")

    def start(self):
        """Start the frame reading thread"""
        self.thread.start()
        return self

    def update(self):
        """Main loop for reading and processing frames"""
        start_time = time.perf_counter()
        grabbed = self.stream.grab()

        while not self.stopped:
            cycle_start = time.perf_counter()

            grabbed, frame = self.stream.retrieve()
            self.frames_read += 1

            self.log_fps()
            processed_frame = self.process_frame(grabbed, frame)
            self.queue.put(processed_frame)

            if self.method == 'feed':
                self.adjust_timing(cycle_start)
            else:
                self.frame_skipping()

        self.stream.release()

    def process_frame(self, grabbed, frame):
        """Process the read frame - returns both resized and original"""
        result = {'camera_status': grabbed}

        if grabbed and frame is not None:
            # Store original frame
            result['original_frame'] = frame
            result['uncropped_base_frame_size'] = frame.shape

            # Crop and resize for processing
            x1, y1, x2, y2 = self.crop_coord_pri
            crop_img = frame[y1:y2, x1:x2]
            result['image'] = cv2.resize(crop_img, (320, 320))
        else:
            result['image'] = None
            result['original_frame'] = None
            self.reconnect()

        return result

    def reconnect(self):
        """Reconnect to the video stream if connection is lost"""
        self.vsq_logger.warning("Reconnecting to video stream...")
        self.stream.release()
        self.stream = cv2.VideoCapture(self.video_input)

    def log_fps(self):
        """Log the current FPS"""
        current_time = time.perf_counter()
        elapsed_time = current_time - self.last_log_time
        if elapsed_time >= 5:
            fps = self.frames_read / elapsed_time
            self.vsq_logger.info(f"Stream FPS:{fps:.2f}")
            self.vsq_logger.info(f"Queue size = {self.queue.qsize()}")
            self.last_log_time = current_time
            self.frames_read = 0

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True
        self.thread.join(2)

    def adjust_timing(self, cycle_start):
        """Adjust timing to maintain target FPS"""
        increment = self.queue.qsize() // self.max_camera_fps
        gap = min(time.perf_counter() - cycle_start - self.target_cycle_time, 0)

        count = 0
        while gap <= 0:
            self.stream.grab()
            time.sleep(1 / self.max_camera_fps)
            count += 1
            gap = time.perf_counter() - cycle_start - self.target_cycle_time

        self.target_cycle_time = max(1 / self.target_fps + increment / 100 - gap, 0)

    def frame_skipping(self):
        """Skip frames in a video"""
        count = 0
        while count < self.skip_frame:
            self.stream.grab()
            time.sleep(1 / self.max_camera_fps)
            count += 1
