import time
import datetime
import logging
import os
import sys
import signal
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager

try:
    from reid_models import ReIDModel
    from reid_processing import process_dwell_time
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    sys.exit(1)

# Constants
EXECUTION_INTERVAL_HOURS = 2
WINDOW_START_HOUR = 8
WINDOW_END_HOUR = 23
SERVICE_BUFFER_SECONDS = 60
LOOP_SLEEP_SECONDS = 60
EXPIRATION_HOURS = 4  # File expiration threshold
MAX_TIME_DIFF_HOURS = 1
SIMILARITY_THRESHOLD = 0.6
MIN_TIME_THRESHOLD = 15  # seconds
MAX_LOAD_AVERAGE = 4
LOAD_WAIT_SECONDS = 30

# Setup logging
log_dir = Path("/tmp/usecases_log/")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'processing_bucket_dwell_age.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration data class"""
    json_folder: Path
    output_dir: Path
    model_path: Path
    store_code: str
    channel_no: str
    serial_id: str

    @classmethod
    def load(cls):
        """Load configuration from file and system"""
        store_code = "Unknown"
        channel_no = "Unknown"
        serial_id = "Unknown"

        try:
            config_path = '/home/celestial/retail_bootstrap_ui/ui/data/footfall_config.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    data = json.load(f)
                store_code = data.get("store_code", "Unknown")
                channel_no = data.get("channelno", "Unknown")

            if os.path.exists('/sys/firmware/devicetree/base/serial-number'):
                serial_id = os.popen('sudo cat /sys/firmware/devicetree/base/serial-number').read().replace('\x00', '').strip()
        except Exception as e:
            logger.error(f"Error loading config: {e}")

        json_folder = Path("/tmp/dwell_time_bbox_json/")
        output_dir = Path("/tmp/dwell_time_jsons")

        json_folder.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            json_folder=json_folder,
            output_dir=output_dir,
            model_path=Path("/home/celestial/detector/models/reid_customnet_x1_0.onnx"),
            store_code=store_code,
            channel_no=channel_no,
            serial_id=serial_id
        )


class ReIDScheduler:
    def __init__(self):
        self.last_execution_time = None
        self.running = True
        self.service_name = "store_manager.service"
        self.config = Config.load()

        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        logger.info(f"ReID Scheduler initialized with store_code={self.config.store_code}, "
                   f"channel_no={self.config.channel_no}, serial_id={self.config.serial_id}")

    def signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.running = False

    def verify_prerequisites(self):
        """Check all required files/directories before starting"""
        checks = {
            'ReID Model': self.config.model_path,
            'JSON Folder': self.config.json_folder,
        }

        all_good = True
        for name, path in checks.items():
            if not path.exists():
                logger.error(f"Missing: {name} at {path}")
                all_good = False

        return all_good

    def is_within_window(self):
        """Check if current time is within operational window"""
        now = datetime.datetime.now()
        return WINDOW_START_HOUR <= now.hour <= WINDOW_END_HOUR

    def should_run(self):
        """Determine if the job should run based on time window and interval"""
        if not self.is_within_window():
            return False

        if self.last_execution_time is None:
            return True

        time_since_last_run = (datetime.datetime.now() - self.last_execution_time).total_seconds()
        return time_since_last_run >= (EXECUTION_INTERVAL_HOURS * 3600)

    def has_files_to_process(self):
        """Check if there are JSON files to process"""
        try:
            files = list(self.config.json_folder.glob('*.json'))
            logger.info(f"Found {len(files)} JSON files in {self.config.json_folder}")
            return len(files) > 0
        except Exception as e:
            logger.error(f"Error checking folder {self.config.json_folder}: {e}")
            return False

    @contextmanager
    def service_paused(self):
        """Context manager to pause and restart the other service"""
        try:
            self.manage_other_service("stop")
            logger.info(f"Waiting {SERVICE_BUFFER_SECONDS} seconds after stopping service...")
            time.sleep(SERVICE_BUFFER_SECONDS)
            yield
        finally:
            logger.info(f"Waiting {SERVICE_BUFFER_SECONDS} seconds before restarting service...")
            time.sleep(SERVICE_BUFFER_SECONDS)
            self.manage_other_service("restart")

    def wait_for_low_load(self, step_name):
        """Block until 1-min load average drops below MAX_LOAD_AVERAGE"""
        while True:
            load = os.getloadavg()[0]
            if load < MAX_LOAD_AVERAGE:
                return
            logger.warning(f"[{step_name}] Load average {load:.2f} >= {MAX_LOAD_AVERAGE}. "
                           f"Waiting {LOAD_WAIT_SECONDS}s...")
            time.sleep(LOAD_WAIT_SECONDS)

    def manage_other_service(self, action):
        """Start/stop/restart the other service"""
        try:
            logger.info(f"Attempting to {action} {self.service_name}...")
            subprocess.run(["systemctl", "--user", action, self.service_name], check=True)
            logger.info(f"Successfully executed: systemctl --user {action} {self.service_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to {action} {self.service_name}. Error code: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while trying to {action} {self.service_name}: {e}")

    def run_reid_matching(self, max_retries=3):
        """Execute the ReID matching process with retry logic"""
        if not self.verify_prerequisites():
            logger.error("Prerequisites check failed. Aborting.")
            return False

        if not self.has_files_to_process():
            logger.info("No files to process.")
            return True

        for attempt in range(max_retries):
            reid_model = None

            try:
                with self.service_paused():
                    logger.info(f"Starting ReID processing (Attempt {attempt + 1}/{max_retries})...")

                    # Step 1: Load ReID model
                    self.wait_for_low_load("Step 1: Load ReID model")
                    logger.info("Step 1: Loading ReID model...")
                    reid_model = ReIDModel(str(self.config.model_path))
                    if not reid_model.load():
                        raise Exception("Failed to load ReID model")

                    # Step 2: Process dwell time
                    self.wait_for_low_load("Step 2: Dwell time matching")
                    logger.info("Step 2: Processing dwell time matching...")
                    current_time = datetime.datetime.now().strftime("%H_%M_%S")
                    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                    output_file = f"{current_date}_{current_time}_{self.config.store_code}_{self.config.channel_no}.json"
                    output_json_path = self.config.output_dir / output_file

                    processed_files = process_dwell_time(
                        reid_model=reid_model,
                        json_folder=str(self.config.json_folder),
                        output_json=str(output_json_path),
                        max_time_diff_hours=MAX_TIME_DIFF_HOURS,
                        sim_threshold=SIMILARITY_THRESHOLD,
                        min_time_threshold=MIN_TIME_THRESHOLD,
                        exclude_staff=True,
                        expiration_hours=EXPIRATION_HOURS,
                        store_code=self.config.store_code,
                        channel_no=self.config.channel_no,
                        serial_id=self.config.serial_id
                    )

                    logger.info(f"Dwell time processing completed. Processed {len(processed_files)} files.")

                    # Unload ReID model to free memory
                    reid_model.unload()
                    reid_model = None

                    # Step 3: Delete all processed files
                    logger.info("Step 3: Cleaning up processed files...")
                    deleted_count = 0
                    for file_path in processed_files:
                        try:
                            file_path_obj = Path(file_path)
                            if file_path_obj.exists():
                                file_path_obj.unlink()
                                deleted_count += 1
                                logger.debug(f"Deleted: {file_path_obj.name}")
                        except Exception as e:
                            logger.error(f"Failed to delete {file_path}: {e}")

                    logger.info(f"Cleanup completed. Deleted {deleted_count}/{len(processed_files)} files.")
                    logger.info("=== All processing completed successfully! ===")
                    return True

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}", exc_info=True)

                if reid_model:
                    reid_model.unload()

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in 30 seconds...")
                    time.sleep(30)
                else:
                    logger.error("All retry attempts failed.")
                    return False

        return False

    def run(self):
        """Main scheduler loop"""
        logger.info(f"ReID Scheduler service started.")
        logger.info(f"Monitoring window: {WINDOW_START_HOUR}:00-{WINDOW_END_HOUR}:00")
        logger.info(f"Execution interval: {EXECUTION_INTERVAL_HOURS} hours")
        logger.info(f"File expiration threshold: {EXPIRATION_HOURS} hours")

        while self.running:
            try:
                if self.should_run():
                    logger.info("Conditions met (Inside Window & Time Elapsed). Executing job...")

                    if self.run_reid_matching():
                        self.last_execution_time = datetime.datetime.now()
                        next_run = self.last_execution_time + datetime.timedelta(hours=EXECUTION_INTERVAL_HOURS)
                        logger.info(f"cycle_completed")
                        logger.info(f"Job finished successfully. Next run allowed after: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        logger.error("Job failed. Will retry in next loop iteration.")

                time.sleep(LOOP_SLEEP_SECONDS)
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt. Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                time.sleep(LOOP_SLEEP_SECONDS)

        logger.info("ReID Scheduler service stopped.")


if __name__ == "__main__":
    scheduler = ReIDScheduler()
    scheduler.run()
