import os
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from reid_models import decode_image_from_base64, cosine_similarity

MAX_LOAD_AVERAGE = 4
LOAD_WAIT_SECONDS = 30


def wait_for_low_load(context=""):
    """Block until 1-min load average drops below MAX_LOAD_AVERAGE"""
    while True:
        load = os.getloadavg()[0]
        if load < MAX_LOAD_AVERAGE:
            return
        print(f"[{context}] Load average {load:.2f} >= {MAX_LOAD_AVERAGE}. Waiting {LOAD_WAIT_SECONDS}s...")
        time.sleep(LOAD_WAIT_SECONDS)


def parse_json_timestamp(date_time_str):
    """Parse timestamp from JSON date_time field"""
    try:
        dt_obj = datetime.strptime(date_time_str, "%Y-%m-%d_%H-%M-%S")
        timestamp = dt_obj.timestamp()
        formatted_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        return timestamp, formatted_str
    except ValueError as e:
        print(f"Error parsing date {date_time_str}: {e}")
        return None, None


def format_time_difference(time_diff):
    """Format time difference in human-readable format"""
    if time_diff is None:
        return "Unknown"
    if time_diff < 60:
        return f"{time_diff:.2f} seconds"
    elif time_diff < 3600:
        minutes = time_diff / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = time_diff / 3600
        return f"{hours:.2f} hours"


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is None:
        return None
    else:
        return obj


def process_dwell_time(reid_model, json_folder, output_json,
                       max_time_diff_hours=1, sim_threshold=0.75,
                       min_time_threshold=15, exclude_staff=True,
                       expiration_hours=4,
                       store_code=None, channel_no=None, serial_id=None):
    """
    Process dwell time matching from JSON folder

    Returns:
        list: Paths of all processed files (matched entries, matched exits, expired files, staff files)
    """

    print(f"=== Starting Dwell Time Processing ===")
    print(f"JSON Folder: {json_folder}")
    print(f"Similarity Threshold: {sim_threshold}")
    print(f"Max Time Diff: {max_time_diff_hours} hours")
    print(f"Expiration Threshold: {expiration_hours} hours")

    json_folder_path = Path(json_folder)
    json_files = list(json_folder_path.glob('*.json'))
    print(f"Found {len(json_files)} JSON files.")

    entries = []
    exits = []
    staff_count = 0
    processed_files = []

    expiration_seconds = expiration_hours * 3600
    current_ts_now = datetime.now().timestamp()

    # Process all JSON files
    for file_idx, j_path in enumerate(json_files):
        if file_idx % 50 == 0:
            wait_for_low_load(f"File extraction {file_idx}/{len(json_files)}")
        filename = j_path.name
        try:
            with open(j_path, 'r') as f:
                data = json.load(f)

            direction = data.get('direction', '')
            date_time_str = data.get('date_time', '')
            image_b64 = data.get('image_bytes', '')
            ob_class = data.get('object_class', '')

            if not image_b64 or not date_time_str:
                continue

            # Parse timestamp from JSON
            ts, ts_str = parse_json_timestamp(date_time_str)
            if ts is None:
                continue

            # Check if file is expired (based on date_time in JSON)
            age_seconds = current_ts_now - ts
            if age_seconds > expiration_seconds:
                print(f"File {filename} is expired ({age_seconds/3600:.1f}h old). Marking for deletion.")
                processed_files.append(str(j_path))
                continue

            # Exclude staff
            if exclude_staff and ob_class == 'staff':
                staff_count += 1
                processed_files.append(str(j_path))
                continue

            # Decode images
            img_cv2, img_pil = decode_image_from_base64(image_b64)
            if img_cv2 is None or img_pil is None:
                continue

            # Extract ReID features
            embedding = reid_model.extract_features(img_pil)
            if embedding is None:
                continue

            record = {
                'filepath': str(j_path),
                'filename': filename,
                'timestamp': ts,
                'datetime_str': ts_str,
                'embedding': embedding,
                'gender': data.get('object_class', None),
                'data': data
            }

            if direction == 'pos':
                entries.append(record)
            elif direction == 'neg':
                exits.append(record)

        except Exception as e:
            print(f"Error reading JSON {j_path}: {e}")

    print(f"Valid Data: {len(entries)} Entries, {len(exits)} Exits. Excluded {staff_count} Staff.")

    # Match entries and exits
    max_time_diff_seconds = max_time_diff_hours * 3600
    all_candidates = []

    print("Generating candidate pairs...")
    for i, entry in enumerate(entries):
        if i % 100 == 0:
            wait_for_low_load(f"Candidate matching entry {i}/{len(entries)}")
        for j, exit_record in enumerate(exits):
            if exit_record['timestamp'] < entry['timestamp']:
                continue

            time_diff = exit_record['timestamp'] - entry['timestamp']
            if time_diff > max_time_diff_seconds or time_diff < min_time_threshold:
                continue

            sim = cosine_similarity(entry['embedding'], exit_record['embedding'])
            if sim >= sim_threshold:
                all_candidates.append({
                    'score': sim,
                    'entry_idx': i,
                    'exit_idx': j,
                    'time_diff': time_diff
                })

    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    print(f"Found {len(all_candidates)} candidates above threshold. Resolving conflicts...")

    # Resolve conflicts (one-to-one matching)
    matches_data = []
    used_entries = set()
    used_exits = set()
    match_counter = 0

    matched_entry_files = set()
    matched_exit_files = set()

    for cand in all_candidates:
        e_idx = cand['entry_idx']
        x_idx = cand['exit_idx']

        if e_idx in used_entries or x_idx in used_exits:
            continue

        used_entries.add(e_idx)
        used_exits.add(x_idx)
        match_counter += 1

        entry = entries[e_idx]
        exit_match = exits[x_idx]
        score = cand['score']
        time_diff = cand['time_diff']

        matches_data.append({
            'match_id': match_counter,
            'gender': entry['gender'],
            'entry_filename': entry['filename'],
            'exit_filename': exit_match['filename'],
            'entry_time_str': entry['datetime_str'],
            'exit_time_str': exit_match['datetime_str'],
            'score': score,
            'time_diff': time_diff
        })

        matched_entry_files.add(entry['filepath'])
        matched_exit_files.add(exit_match['filepath'])

    print(f"Found {len(matches_data)} valid matches.")

    # Add matched files to processed list
    processed_files.extend(list(matched_entry_files))
    processed_files.extend(list(matched_exit_files))
    processed_files = list(set(processed_files))  # Remove duplicates

    # Save dwell time output JSON
    save_dwell_time_json(
        matches_data=matches_data,
        staff_count=staff_count,
        output_file=output_json,
        store_code=store_code,
        channel_no=channel_no,
        serial_id=serial_id
    )

    print(f"=== Dwell Time Processing Complete ===")
    print(f"Total processed files: {len(processed_files)}")

    return processed_files


def save_dwell_time_json(matches_data, staff_count, output_file,
                         store_code=None, channel_no=None, serial_id=None):
    """Save dwell time matches to JSON file"""

    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")

    final_matches = []

    for m in matches_data:
        time_diff = m['time_diff']

        match_entry = {
            "match_id": m.get("match_id"),
            "gender": m.get("gender"),
            "entry_filename": m['entry_filename'],
            "entry_time": m['entry_time_str'],
            "exit_filename": m['exit_filename'],
            "exit_time": m['exit_time_str'],
            "similarity_score": round(convert_numpy_types(m['score']), 4),
            "time_difference_seconds": round(convert_numpy_types(time_diff), 2),
            "time_difference_formatted": format_time_difference(time_diff)
        }
        final_matches.append(match_entry)

    summary = {
        "date_time": f"{current_date} {current_time}",
        "store_code": f"{store_code}" if store_code else "Unknown",
        "camera_no": int(channel_no) if channel_no not in (None, "Unknown") else None,
        "device_serial_id": f"{serial_id}" if serial_id else "Unknown",
        "staff_stats": {
            "total_staff_excluded": staff_count
        }
    }

    result = {
        **summary,
        "matches": final_matches
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Dwell time results saved to {output_file}")


def process_age_groups(age_model, file_list, json_folder, age_output_dir,
                       store_code=None, channel_no=None, serial_id=None):
    """
    Process age group prediction for given file list

    Args:
        age_model: Loaded AgeModel instance
        file_list: List of file paths to process
        json_folder: Path to JSON folder
        age_output_dir: Directory to save age JSONs
        store_code, channel_no, serial_id: Metadata
    """

    print(f"=== Starting Age Group Processing ===")
    print(f"Files to process: {len(file_list)}")

    os.makedirs(age_output_dir, exist_ok=True)

    age_processed_count = 0

    for age_idx, file_path in enumerate(file_list):
        if age_idx % 50 == 0:
            wait_for_low_load(f"Age processing {age_idx}/{len(file_list)}")
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            print(f"File not found: {file_path}")
            continue

        filename = file_path_obj.name

        try:
            with open(file_path_obj, 'r') as f:
                data = json.load(f)

            direction = data.get('direction', '')
            image_b64 = data.get('image_bytes', '')

            # Only process 'pos' direction files
            if direction != 'pos':
                continue

            if not image_b64:
                continue

            # Decode image
            img_cv2, img_pil = decode_image_from_base64(image_b64)
            if img_cv2 is None:
                continue

            # Predict age
            predicted_age_group = age_model.predict_age(img_cv2)

            # Save age result
            age_filename = f"age_{filename}"
            age_json_path = os.path.join(age_output_dir, age_filename)

            age_result = {
                "filename": filename,
                "age_group": predicted_age_group,
                "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "store_code": f"{store_code}" if store_code else "Unknown",
                "camera_no": int(channel_no) if channel_no not in (None, "Unknown") else None,
                "device_serial_id": f"{serial_id}" if serial_id else "Unknown",
            }

            with open(age_json_path, 'w') as af:
                json.dump(age_result, af, indent=2)

            age_processed_count += 1

        except Exception as e:
            print(f"Error processing age for {filename}: {e}")

    print(f"=== Age Group Processing Complete ===")
    print(f"Processed {age_processed_count} files for age prediction.")
