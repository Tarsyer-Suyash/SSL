#!/usr/bin/env python3
"""
Hotspot Cleanliness / Arrangement Monitoring System
=====================================================
Use case : Monitor retail display hotspots (tables, shelves, racks)
           for proper merchandise arrangement.
           Crops each configured ROI and sends to Qwen Vision.
           If merchandise is messy/disarranged → saves alert JSON.

Flow per cycle:
  1. Load config from hotspot.json
  2. For each camera → capture frame via ffmpeg (with -ss warm-up skip)
  3. For each ROI defined for that camera → crop the ROI region
  4. Save crop to /tmp/hotspot_crops/
  5. Send crop to Qwen Vision → ask if merchandise is properly arranged
  6. If violation → save alert JSON + image to /tmp/alert_json/
  7. Per-ROI cooldown applies

Config  : /home/celestial/retail_bootstrap_ui/ui/data/hotspot.json
Logs    : /tmp/usecases_log/hotspot_YYYYMMDD.log
Crops   : /tmp/hotspot_crops/
Alerts  : /tmp/alert_json/

hotspot.json schema:
{
  "store_code": "STORE-001",
  "device_serial_id": "auto",
  "capture_interval_sec": 3600,
  "alert_cooldown_sec": 3600,
  "capture_ss_sec": 3,                      <-- optional (global ss, seconds)
  "cameras": [
    {
      "camera_no": 1,
      "rtsp_url": "rtsp://admin:pass@192.168.1.202:554/Streaming/Channels/101",
      "capture_ss_sec": 3,                  <-- optional (per-camera override)
      "hotspots": [
        {
          "hotspot_id": "HS001",
          "label": "T-Shirt Table Center",
          "roi": [[600,250],[1040,250],[1040,660],[600,660]],
          "merchandise_type": "folded clothes on display table"
        }
      ]
    }
  ]
}
"""

import os
import cv2
import time
import json
import base64
import logging
import subprocess
from datetime import datetime

import numpy as np
import requests

# ══════════════════════════════════════════════════════════════
#  QWEN VISION API
# ══════════════════════════════════════════════════════════════
QWEN_API_URL       = "https://kwen.tarsyer.com/v1/chat/completions"
QWEN_API_KEY       = os.environ.get("QWEN_API_KEY", "")
QWEN_TARGET_SERVER = "BK"

# ══════════════════════════════════════════════════════════════
#  CONFIG PATH
# ══════════════════════════════════════════════════════════════
CONFIG_PATH = "/home/celestial/retail_bootstrap_ui/ui/data/hotspot.json"

# ══════════════════════════════════════════════════════════════
#  FRAME CAPTURE SETTINGS
# ══════════════════════════════════════════════════════════════
# ffmpeg -ss : seconds of stream to skip/discard before grabbing the frame.
# The first frames of an RTSP stream are often grey/green/half-decoded
# (decoder hasn't received a full keyframe yet). Skipping a few seconds
# lets the decoder settle so the captured frame has proper colour.
DEFAULT_CAPTURE_SS_SEC = 3

# Retry capture if the frame still looks grayscale
MAX_CAPTURE_RETRIES    = 3
# Mean per-pixel channel difference below this => treat frame as grayscale
GRAY_CHANNEL_DIFF_THRESH = 2.0

# ══════════════════════════════════════════════════════════════
#  FILE PATHS
# ══════════════════════════════════════════════════════════════
LOG_DIR   = "/tmp/usecases_log"
CROP_DIR  = "/tmp/hotspot_crops"
ALERT_DIR = "/tmp/alert_json"

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(CROP_DIR,  exist_ok=True)
os.makedirs(ALERT_DIR, exist_ok=True)

log_file = os.path.join(
    LOG_DIR, f"hotspot_{datetime.now().strftime('%Y%m%d')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HotspotMonitor")


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def get_device_serial() -> str:
    try:
        serial = os.popen(
            "sudo cat /sys/firmware/devicetree/base/serial-number"
        ).read().replace("\x00", "").strip()
        return serial if serial else "unknown"
    except Exception:
        return "unknown"


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
#  FRAME CAPTURE  (ffmpeg with -ss)
# ══════════════════════════════════════════════════════════════
def is_grayscale_frame(frame) -> bool:
    """True if the frame has (almost) no colour information."""
    if frame is None or frame.ndim < 3 or frame.shape[2] < 3:
        return True
    f = frame.astype(np.int16)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    diff = (np.abs(b - g).mean() + np.abs(g - r).mean()) / 2.0
    return diff < GRAY_CHANNEL_DIFF_THRESH


def extract_frame(rtsp_url: str, cam_no, ss_sec: float = DEFAULT_CAPTURE_SS_SEC):
    """
    Grab one frame from the RTSP stream.

    -ss is placed AFTER -i (output option) on purpose: for a live RTSP
    stream ffmpeg then decodes and discards the first `ss_sec` seconds and
    grabs the next frame, which gives the decoder time to receive a full
    keyframe. (Placing -ss before -i tries to seek the input, which does
    not work for live streams.)
    """
    image_path = f"/tmp/hotspot_cam{cam_no}.jpg"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-analyzeduration", "5000000",
        "-probesize",       "5000000",
        "-i", rtsp_url,
        "-ss", str(ss_sec),          # <-- skip warm-up frames
        "-an",
        "-vframes", "1",
        "-q:v", "2",
        "-y", image_path
    ]

    last_frame = None
    for attempt in range(1, MAX_CAPTURE_RETRIES + 1):
        try:
            if os.path.exists(image_path):
                os.remove(image_path)

            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=30 + int(ss_sec)
            )
            if result.returncode == 0 and os.path.exists(image_path):
                frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if frame is not None:
                    last_frame = frame
                    if is_grayscale_frame(frame):
                        logger.warning(
                            f"[CAM{cam_no}] Frame looks grayscale "
                            f"(attempt {attempt}/{MAX_CAPTURE_RETRIES}, ss={ss_sec}s)"
                        )
                        # increase skip time slightly on each retry
                        cmd[cmd.index("-ss") + 1] = str(ss_sec + attempt * 2)
                        continue
                    logger.info(
                        f"[CAM{cam_no}] Frame captured {frame.shape}  ss={ss_sec}s"
                    )
                    return frame
            else:
                logger.error(
                    f"[CAM{cam_no}] ffmpeg failed (attempt {attempt}): "
                    f"{result.stderr[-200:]}"
                )
        except Exception as e:
            logger.error(f"[CAM{cam_no}] ffmpeg exception (attempt {attempt}): {e}")

    if last_frame is not None:
        # Could be a genuine night/IR-mode grayscale scene — still use it.
        logger.warning(
            f"[CAM{cam_no}] Frame still grayscale after {MAX_CAPTURE_RETRIES} "
            f"attempts — using it anyway (camera may be in IR/night mode)."
        )
        return last_frame
    return None


# ══════════════════════════════════════════════════════════════
#  ROI CROP
# ══════════════════════════════════════════════════════════════
def crop_roi(frame, roi_points, padding: int = 10):
    """Crop the bounding rect of the ROI polygon with padding."""
    pts        = np.array(roi_points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    img_h, img_w = frame.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    return frame[y1:y2, x1:x2]


def draw_roi_on_frame(frame, roi_points, color=(0, 255, 0), label=None):
    """Draw semi-transparent ROI polygon on frame."""
    vis     = frame.copy()
    pts     = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))
    overlay = vis.copy()
    cv2.fillPoly(overlay, [pts], color)
    vis = cv2.addWeighted(overlay, 0.20, vis, 0.80, 0)
    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
    if label:
        x, y, _, _ = cv2.boundingRect(np.array(roi_points, dtype=np.int32))
        cv2.putText(vis, label, (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


# ══════════════════════════════════════════════════════════════
#  QWEN PROMPT
# ══════════════════════════════════════════════════════════════
def build_prompt(merchandise_type: str, label: str) -> str:
    return f"""You are a retail store visual merchandising inspector analyzing a CCTV image.

This cropped image shows a specific retail display area: "{label}".
The merchandise type at this hotspot is: {merchandise_type}.

Your task: Determine whether the merchandise display is PROPERLY ARRANGED or MESSY/DISARRANGED.

What counts as PROPERLY ARRANGED:
  - Items are neatly folded, stacked, or hung in an organized manner
  - Display looks tidy, consistent, and presentable to customers
  - Items are properly placed within the display area
  - No visible gaps, fallen items, or scattered merchandise

What counts as MESSY / DISARRANGED (violation):
  - Clothes unfolded, crumpled, or thrown randomly
  - Items fallen off shelves or racks
  - Merchandise scattered or piled haphazardly
  - Visible large gaps where stock has been removed without reorganizing
  - Display looks noticeably untidy compared to a well-maintained store

IMPORTANT rules:
  - Minor imperfections (one slightly unaligned item) do NOT count as a violation
  - Only flag as violation if the display is CLEARLY and NOTICEABLY messy
  - If image is too dark, blurry, or the display area is not clearly visible → set confidence=low, is_arranged=true (no false alert)
  - Only judge what is clearly visible. Do not assume.

Respond ONLY with this exact JSON — no markdown, no extra text:
{{
  "display_visible": true or false,
  "is_arranged": true or false,
  "violation_detected": true or false,
  "severity": "none" or "minor" or "moderate" or "severe",
  "violation_details": "Describe exactly what is wrong with the display. Write 'None' if properly arranged.",
  "confidence": "high" or "medium" or "low"
}}

- violation_detected=true ONLY if display is clearly messy or disarranged.
- severity: none=clean, minor=borderline (no alert), moderate/severe=alert required.
- confidence=low → violation_detected=false (no alert).
- When in doubt → do NOT flag."""


# ══════════════════════════════════════════════════════════════
#  QWEN API CALL
# ══════════════════════════════════════════════════════════════
def analyze_with_qwen(image_b64: str, cam_no, hotspot_id: str,
                      merchandise_type: str, label: str) -> dict:
    prompt  = build_prompt(merchandise_type, label)
    headers = {
        "Content-Type": "application/json",
        "X-API-Key":    QWEN_API_KEY
    }
    payload = {
        "target_server": QWEN_TARGET_SERVER,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }}
                ]
            }
        ],
        "max_tokens": 512
    }
    try:
        logger.info(
            f"[CAM{cam_no}][{hotspot_id}] Sending crop to Qwen ..."
        )
        resp = requests.post(
            QWEN_API_URL, headers=headers, json=payload, timeout=60
        )
        resp.raise_for_status()

        raw = resp.json()["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw   = parts[1] if len(parts) > 1 else parts[0]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        logger.info(
            f"[CAM{cam_no}][{hotspot_id}] Qwen result → {result}"
        )
        return result

    except requests.exceptions.RequestException as e:
        logger.error(f"[CAM{cam_no}][{hotspot_id}] Qwen API error: {e}")
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"[CAM{cam_no}][{hotspot_id}] Qwen parse error: {e}")
    return None


# ══════════════════════════════════════════════════════════════
#  ALERT SAVER
# ══════════════════════════════════════════════════════════════
def save_alert(cam_no, hotspot_id, label, crop,
               qwen_result, device_serial, store_code) -> str:
    now    = datetime.now()
    ts_str = now.strftime("%Y%m%d_%H%M%S")

    severity = qwen_result.get("severity",          "unknown")
    details  = qwen_result.get("violation_details", "NA") or "NA"
    status   = f"VIOLATION: Hotspot display messy/disarranged — severity={severity}"

    _, buf    = cv2.imencode(".jpg", crop)
    image_b64 = base64.b64encode(buf).decode("utf-8")

    alert = {
        "date_time":        now.strftime("%Y-%m-%d %H:%M:%S"),
        "device_serial_id": device_serial,
        "store_code":       store_code,
        "camera_no":        cam_no,
        "explanation":      details,
        "alert_type":       "hotspot",
        "hotspot_id":       hotspot_id,
        "hotspot_label":    label,
        "severity":         severity,
        "response":         status,
        "image_byte_str":   image_b64
    }

    json_path = os.path.join(
        ALERT_DIR,
        f"alert_{ts_str}_cam{cam_no}_{hotspot_id}_hotspot.json"
    )
    with open(json_path, "w") as f:
        json.dump(alert, f, indent=2)
    logger.info(f"[CAM{cam_no}][{hotspot_id}] Alert JSON  → {json_path}")

    img_path = os.path.join(
        CROP_DIR,
        f"{ts_str}_cam{cam_no}_{hotspot_id}_alert.jpg"
    )
    cv2.imwrite(img_path, crop)
    logger.info(f"[CAM{cam_no}][{hotspot_id}] Alert image → {img_path}")

    return json_path


# ══════════════════════════════════════════════════════════════
#  MAIN MONITOR
# ══════════════════════════════════════════════════════════════
class HotspotMonitor:

    def __init__(self):
        cfg                   = load_config(CONFIG_PATH)
        self.store_code       = cfg.get("store_code",           "UNKNOWN")
        self.capture_interval = cfg.get("capture_interval_sec", 3600)
        self.cooldown_sec     = cfg.get("alert_cooldown_sec",   3600)
        self.capture_ss       = cfg.get("capture_ss_sec", DEFAULT_CAPTURE_SS_SEC)
        self.cameras          = cfg.get("cameras",              [])

        # device serial
        serial = cfg.get("device_serial_id", "auto")
        self.device_serial = get_device_serial() if serial == "auto" else serial

        # Per (camera_no, hotspot_id) cooldown tracker
        self.last_alert = {}
        for cam in self.cameras:
            for hs in cam.get("hotspots", []):
                key = (cam["camera_no"], hs["hotspot_id"])
                self.last_alert[key] = 0

        logger.info("=" * 60)
        logger.info("  Hotspot Arrangement Monitoring System — Starting Up")
        logger.info(f"  Config   : {CONFIG_PATH}")
        for cam in self.cameras:
            safe = cam["rtsp_url"].split("@")[-1]
            logger.info(
                f"  CAM{cam['camera_no']} : rtsp://***@{safe}  "
                f"hotspots={len(cam.get('hotspots', []))}  "
                f"ss={cam.get('capture_ss_sec', self.capture_ss)}s"
            )
            for hs in cam.get("hotspots", []):
                logger.info(
                    f"    [{hs['hotspot_id']}] {hs['label']} — "
                    f"{hs.get('merchandise_type', 'general display')}"
                )
        logger.info(f"  Store    : {self.store_code}  Device: {self.device_serial}")
        logger.info(f"  Qwen     : {QWEN_API_URL}  target={QWEN_TARGET_SERVER}")
        logger.info(f"  Interval : {self.capture_interval}s  Cooldown: {self.cooldown_sec}s")
        logger.info("=" * 60)

    def process_camera(self, cam: dict):
        cam_no   = cam["camera_no"]
        rtsp_url = cam["rtsp_url"]
        hotspots = cam.get("hotspots", [])
        ss_sec   = cam.get("capture_ss_sec", self.capture_ss)

        if not hotspots:
            logger.warning(f"[CAM{cam_no}] No hotspots configured — skipping.")
            return

        # Check if any hotspot needs processing (not all in cooldown)
        needs_frame = False
        for hs in hotspots:
            key       = (cam_no, hs["hotspot_id"])
            remaining = self.cooldown_sec - (time.time() - self.last_alert[key])
            if remaining <= 0:
                needs_frame = True
                break

        if not needs_frame:
            logger.info(f"[CAM{cam_no}] All hotspots in cooldown — skipping frame capture.")
            return

        # Capture frame (with -ss warm-up skip)
        frame = extract_frame(rtsp_url, cam_no, ss_sec)
        if frame is None:
            logger.warning(f"[CAM{cam_no}] Frame capture failed — skipping.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save annotated frame with all ROIs drawn
        vis = frame.copy()
        for hs in hotspots:
            vis = draw_roi_on_frame(
                vis, hs["roi"],
                color=(0, 255, 0),
                label=hs["hotspot_id"]
            )
        raw_path = os.path.join(CROP_DIR, f"{ts}_cam{cam_no}_frame.jpg")
        cv2.imwrite(raw_path, vis)
        logger.info(f"[CAM{cam_no}] Annotated frame saved → {raw_path}")

        # Process each hotspot
        for hs in hotspots:
            hotspot_id       = hs["hotspot_id"]
            label            = hs.get("label",            hotspot_id)
            roi_points       = hs["roi"]
            merchandise_type = hs.get("merchandise_type", "retail merchandise display")

            key       = (cam_no, hotspot_id)
            remaining = self.cooldown_sec - (time.time() - self.last_alert[key])
            if remaining > 0:
                logger.info(
                    f"[CAM{cam_no}][{hotspot_id}] Cooldown: "
                    f"{int(remaining // 60)}m {int(remaining % 60)}s left."
                )
                continue

            # Crop ROI
            crop = crop_roi(frame, roi_points)
            if crop.size == 0:
                logger.warning(
                    f"[CAM{cam_no}][{hotspot_id}] Empty crop — skipping."
                )
                continue

            # Save crop
            crop_path = os.path.join(
                CROP_DIR, f"{ts}_cam{cam_no}_{hotspot_id}_crop.jpg"
            )
            cv2.imwrite(crop_path, crop)
            logger.info(
                f"[CAM{cam_no}][{hotspot_id}] Crop saved → {crop_path}  "
                f"size={crop.shape}"
            )

            # Send to Qwen
            _, buf    = cv2.imencode(".jpg", crop)
            image_b64 = base64.b64encode(buf).decode("utf-8")

            result = analyze_with_qwen(
                image_b64, cam_no, hotspot_id,
                merchandise_type, label
            )
            if result is None:
                logger.warning(
                    f"[CAM{cam_no}][{hotspot_id}] No Qwen result — skipping."
                )
                continue

            display_visible = result.get("display_visible", True)
            is_arranged     = result.get("is_arranged",     True)
            violation       = result.get("violation_detected", False)
            severity        = result.get("severity",        "none")
            confidence      = result.get("confidence",      "unknown")
            details         = result.get("violation_details", "None")

            logger.info(
                f"[CAM{cam_no}][{hotspot_id}] "
                f"display_visible={display_visible}  "
                f"is_arranged={is_arranged}  "
                f"violation={violation}  "
                f"severity={severity}  "
                f"confidence={confidence}"
            )

            # Alert on moderate/severe only (skip minor)
            if violation and severity in ("moderate", "severe"):
                logger.warning(
                    f"[CAM{cam_no}][{hotspot_id}] HOTSPOT VIOLATION — "
                    f"severity={severity}  details={details}"
                )
                save_alert(
                    cam_no, hotspot_id, label, crop,
                    result, self.device_serial, self.store_code
                )
                self.last_alert[key] = time.time()
                logger.info(
                    f"[CAM{cam_no}][{hotspot_id}] Alert saved. "
                    f"Cooldown started."
                )
            elif violation and severity == "minor":
                logger.info(
                    f"[CAM{cam_no}][{hotspot_id}] Minor issue — "
                    f"no alert (below threshold)."
                )
            else:
                logger.info(
                    f"[CAM{cam_no}][{hotspot_id}] Display properly arranged — "
                    f"no alert."
                )

    def run(self):
        while True:
            load_avg = os.getloadavg()[0]
            if load_avg > 3.5:
                logger.warning(
                    f"High load {load_avg:.2f} > 3.5 — sleeping 10 min ..."
                )
                time.sleep(600)
                continue

            for cam in self.cameras:
                try:
                    self.process_camera(cam)
                except Exception as e:
                    logger.exception(
                        f"[CAM{cam['camera_no']}] Unexpected error: {e}"
                    )

            logger.info(
                f"Cycle complete. Sleeping {self.capture_interval}s ..."
            )
            time.sleep(self.capture_interval)


if __name__ == "__main__":
    monitor = HotspotMonitor()
    monitor.run()
