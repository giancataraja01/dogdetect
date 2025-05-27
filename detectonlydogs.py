import jetson.inference
import jetson.utils

# Load the object detection model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")
display = jetson.utils.glDisplay()
font = jetson.utils.cudaFont()  # Create a font object for drawing text

# Ensure the file exists with an initial value
with open("detection_logs.txt", "w") as log_file:
    log_file.write("false\n")

while display.IsOpen():
    img, width, height = camera.CaptureRGBA()
    # Prevent automatic overlay drawing
    detections = net.Detect(img, width, height, overlay="none")

    dog_detected = False

    # Draw only dog bounding boxes and labels
    for detection in detections:
        class_desc = net.GetClassDesc(detection.ClassID).lower()
        if class_desc == "dog":
            dog_detected = True
            jetson.utils.cudaDrawRect(
                img,
                (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)),
                (255, 255, 0, 255)
            )
            # Draw text label using cudaFont
            font.Overlay(img, "dog", int(detection.Left), int(detection.Top) - 20, (255, 255, 0, 255), width)

    # Update the log file based on detection
    with open("detection_logs.txt", "w") as log_file:
        log_file.write("true\n" if dog_detected else "false\n")

    display.RenderOnce(img, width, height)
    display.SetTitle("Dog Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
