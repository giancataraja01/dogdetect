import os
import time
import psutil
from datetime import timedelta
import firebase_admin
from firebase_admin import credentials, db

# ---------- CONFIGURATION ----------
FIREBASE_CRED_PATH = 'project8-b295f-4cd41be927cd.json'
DATABASE_URL = 'https://project8-b295f-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Update this

# ---------- INITIALIZE FIREBASE ----------
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred, {
        'databaseURL': DATABASE_URL
    })

# ---------- SYSTEM INFO FUNCTIONS ----------
def get_uptime():
    with open('/proc/uptime', 'r') as f:
        uptime_seconds = float(f.readline().split()[0])
        return str(timedelta(seconds=uptime_seconds))

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    mem = psutil.virtual_memory()
    return {
        'total_mb': round(mem.total / (1024 ** 2), 2),
        'used_mb': round(mem.used / (1024 ** 2), 2),
        'percent': mem.percent
    }

def get_temperature():
    temp_path = '/sys/devices/virtual/thermal/thermal_zone0/temp'
    try:
        with open(temp_path, 'r') as f:
            temp_milli = int(f.read().strip())
            return temp_milli / 1000.0
    except FileNotFoundError:
        return None

# ---------- MAIN FUNCTION ----------
def upload_status_to_firebase():
    status = {
        'uptime': get_uptime(),
        'cpu_usage_percent': get_cpu_usage(),
        'memory_usage': get_memory_usage(),
        'temperature_celsius': get_temperature(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    ref = db.reference('jetson_nano/status')
    ref.set(status)
    print("Status uploaded to Firebase:")
    print(status)

# ---------- RUN ----------
if __name__ == '__main__':
    try:
        while True:
            upload_status_to_firebase()
            time.sleep(5)
    except KeyboardInterrupt:
        print("Stopped by user.")
