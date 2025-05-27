import jetson.inference
import jetson.utils

# Load the object detection model once
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Use videoSource/videoOutput for lower memory usage and simpler pipeline
camera = jetson.utils.videoSource("/dev/video0")  # Default: 1280x720
display = jetson.utils.videoOutput("display://0")  # Display output
font = jetson.utils.cudaFont()  # Create font object once

# Pre-open the log file in write mode, keep the handle open for less I/O overhead
log_file = open("detection_logs.txt", "w")
log_file.write("false\n")
log_file.flush()

try:
    while display.IsStreaming():
        img = camera.Capture()
        if img is None:
            continue

        detections = net.Detect(img, overlay="none")
        dog_detected = False

        for detection in detections:
            class_desc = net.GetClassDesc(detection.ClassID).lower()
            if class_desc == "dog":
                dog_detected = True
                jetson.utils.cudaDrawRect(
                    img,
                    (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)),
                    (255, 255, 0, 255)
                )
                font.Overlay(
                    "dog",
                    int(detection.Left),
                    max(0, int(detection.Top) - 20),
                    (255, 255, 0, 255),
                    img
                )
        # Write only if detection state changed, minimizing disk I/O
        log_state = "true\n" if dog_detected else "false\n"
        log_file.seek(0)
        log_file.write(log_state)
        log_file.truncate()
        log_file.flush()

        display.Render(img)
        display.SetTitle("Dog Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
finally:
    log_file.close()
