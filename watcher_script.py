from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os

class LogFileHandler(FileSystemEventHandler):
    def __init__(self, file_path):
        self.file_path = os.path.abspath(file_path)
        self.last_size = 0  # Track last read size

    def on_modified(self, event):
        if os.path.abspath(event.src_path) == self.file_path:
            with open(self.file_path, 'r') as file:
                file.seek(self.last_size)
                new_data = file.read()
                if new_data:
                    print(new_data, end='', flush=True)
                self.last_size = file.tell()

def monitor_log_file(file_path):
    print(f"[INFO] Monitoring '{file_path}'... (Press Ctrl+C to stop)")
    if not os.path.exists(file_path):
        print(f"[WARNING] File '{file_path}' does not exist. Creating it.")
        open(file_path, 'a').close()

    event_handler = LogFileHandler(file_path)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(os.path.abspath(file_path)), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    monitor_log_file("detection_logs.txt")
