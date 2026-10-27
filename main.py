# main.py
import sys
sys.path.insert(0, '/home/celestial/opencv/build/lib/python3')
from config import *


while True:
    current_time = time.time()

    camera_dict = detector.camera_vsq.read()
    if not camera_dict['camera_status']:
        continue

    resized_frame = camera_dict['image']
    original_frame = camera_dict['original_frame']
    frame_counter += 1
    counter += 1

    # Handle FPS calculation
    fps_result = detector.handle_fps_calculation(
        current_time, next_fps_time, counter, start_time)
    if fps_result:
        fps, counter, next_fps_time, start_time, load_average, ram, swap_memory, cpu_temp, cpu_utilization = fps_result

    # Handle JSON saving
    json_result = detector.handle_json_saving(
        current_time, next_json_save, old_dict,
        interval_based_detection_count, fps)
    if json_result:
        old_dict, next_json_save = json_result

    # Handle motion detection
    motion_confirm, thresh = detector.handle_motion_detection(
        motion_confirm, base_frame, resized_frame)

    # Process detections (now includes original_frame for bbox saving)
    (objects, object_start_time, motion_confirm,
     base_frame, interval_based_detection_count, imp_bbox_lists, imp_scores_lists) = detector.process_detections(
        motion_confirm, resized_frame, original_frame, interval_based_detection_count, base_frame)

    # Save gate activity
    # gate_activity.check_and_save_gate_activity(original_frame)

    # Handle debug display
    should_break, insights_img = detector.handle_debug_display(
        DEBUG_SHOW, resized_frame, objects, imp_bbox_lists, imp_scores_lists,
        motion_confirm, load_average, ram,
        swap_memory, cpu_temp, cpu_utilization, fps, thresh)
    if should_break:
        break

    if VIDEO_WRITE and insights_img is not None:
        detector.video_recorder.handle_video_recording(insights_img)

# Cleanup
if VIDEO_WRITE:
    detector.video_recorder.close()

detector.camera_vsq.stop()
cv2.destroyAllWindows()
