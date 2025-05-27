import jetson.inference
import jetson.utils

# Load the object detection model once
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Use videoSource/videoOutput for lower memory usage and simpler pipeline
camera = jetson.utils.videoSource("/dev/video0")  # Default: 1280x720
display = jetson.utils.videoOutput("display://0")  # Display output
font = jetson.utils.cudaFont()  # Create font object once

# Open the log file in write mode, keep the handle open
log_file = open("detection_logs.txt", "w")
last_state = None  # Track previous detection state to minimize disk I/O

try:
    while display.IsStreaming() and camera.IsStreaming():
        img = camera.Capture()
        if img is None:
            continue

        detections = net.Detect(img, overlay="none")
        dog_detected = False

        for detection in detections:
            class_desc = net.GetClassDesc(detection.ClassID).lower()
            if class_desc == "dog":
                dog_detected = True
                # Draw bounding box for dog
                jetson.utils.cudaDrawRect(
                    img,
                    (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)),
                    (255, 255, 0, 255)
                )
                # Draw label for dog (make sure coordinates are within image)
                label_x = max(0, int(detection.Left))
                label_y = max(0, int(detection.Top) - 20)
                font.Overlay(
                    "dog",
                    label_x,
                    label_y,
                    (255, 255, 0, 255),
                    img
                )

        # Write to log only if state changes
        state = "true\n" if dog_detected else "false\n"
        if state != last_state:
            log_file.seek(0)
            log_file.write(state)
            log_file.truncate()
            log_file.flush()
            last_state = state

        display.Render(img)
        display.SetTitle("Dog Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
finally:
    log_file.close()
