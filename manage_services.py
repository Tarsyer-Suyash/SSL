import json
import subprocess
import time
from datetime import datetime

CONFIG_FILE = "/home/celestial/retail_bootstrap_ui/ui/data/manage_services.json"
LOG_FILE = "/tmp/processing_services.log"

def log_message(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}\n"
    
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry)
    
    print(log_entry.strip())

def load_config():
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def is_service_running(service_name):
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', service_name],
            capture_output=True,
            text=True
        )
        return result.stdout.strip() == 'active'
    except Exception as e:
        log_message(f"Error checking {service_name}: {e}")
        return False

def start_service(service_name):
    try:
        subprocess.run(['sudo', 'systemctl', 'start', service_name], check=True)
        log_message(f"Started {service_name}")
    except Exception as e:
        log_message(f"Error starting {service_name}: {e}")

def stop_service(service_name):
    try:
        subprocess.run(['sudo', 'systemctl', 'stop', service_name], check=True)
        log_message(f"Stopped {service_name}")
    except Exception as e:
        log_message(f"Error stopping {service_name}: {e}")

def is_current_time_in_range(time_range):
    start_time, end_time = time_range.split('-')
    start_hour, start_min = map(int, start_time.split(':'))
    end_hour, end_min = map(int, end_time.split(':'))
    
    now = datetime.now()
    current_minutes = now.hour * 60 + now.minute
    start_minutes = start_hour * 60 + start_min
    end_minutes = end_hour * 60 + end_min
    
    # Handle midnight crossing
    if end_minutes < start_minutes:
        return current_minutes >= start_minutes or current_minutes <= end_minutes
    else:
        return start_minutes <= current_minutes <= end_minutes

def manage_services():
    config = load_config()
    services = config['services']
    ranges = config['range']
    
    for service, time_range in zip(services, ranges):
        is_running = is_service_running(service)
        should_run = is_current_time_in_range(time_range)
        
        if should_run and not is_running:
            log_message(f"{service} should be running but isn't. Starting...")
            start_service(service)
        elif not should_run and is_running:
            log_message(f"{service} should not be running but is. Stopping...")
            stop_service(service)
        else:
            log_message(f"{service} is in correct state")

if __name__ == "__main__":
    log_message("System service manager started")
    
    while True:
        try:
            log_message("=== Checking services ===")
            manage_services()
        except Exception as e:
            log_message(f"Error in main loop: {e}")
        
        time.sleep(60)
