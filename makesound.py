import Jetson.GPIO as GPIO
import time

# GPIO setup
TWEETER_PIN = 7  # Example: Pin 33 = GPIO13
GPIO.setmode(GPIO.BOARD)
GPIO.setup(TWEETER_PIN, GPIO.OUT)

# Initialize tweeter state
tweeter_is_on = False

# Read status from text file
def read_status(filename="detection_logs.txt"):
    try:
        with open(filename, "r") as file:
            content = file.read().strip().lower()
            return content == "true"
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return False

print("Monitoring detection_logs.txt... Press Ctrl+C to stop.")

try:
    while True:
        status = read_status()

        if status and not tweeter_is_on:
            GPIO.output(TWEETER_PIN, GPIO.HIGH)
            tweeter_is_on = True
            print("Tweeter ON")
        elif not status and tweeter_is_on:
            GPIO.output(TWEETER_PIN, GPIO.LOW)
            tweeter_is_on = False
            print("Tweeter OFF")

        time.sleep(1)  # Check every 1 second

except KeyboardInterrupt:
    print("Stopping...")

finally:
    GPIO.output(TWEETER_PIN, GPIO.LOW)
    GPIO.cleanup()
    print("GPIO cleaned up. Tweeter OFF.")
