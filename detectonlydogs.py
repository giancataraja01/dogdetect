import jetson.inference
import jetson.utils

# Load the object detection model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")
display = jetson.utils.glDisplay()

# Ensure the file exists with an initial value
with open("detection_logs.txt", "w") as log_file:
    log_file.write("false\n")

while display.IsOpen():
    img, width, height = camera.CaptureRGBA()
    # Set overlay=0 to prevent automatic bounding box drawing
    detections = net.Detect(img, width, height, overlay="none")

    dog_detected = False

    # Draw only dog bounding boxes
    for detection in detections:
        class_desc = net.GetClassDesc(detection.ClassID).lower()
        if class_desc == "dog":
            dog_detected = True
            # Draw bounding box and label only for dogs
            jetson.utils.cudaDrawRect(
                img,
                (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)),
                (255, 255, 0, 255)
            )
            jetson.utils.cudaDrawText(
                img,
                "dog",
                int(detection.Left),
                int(detection.Top) - 20,
                (255, 255, 0, 255),
                width
            )

    # Update the log file based on detection
    with open("detection_logs.txt", "w") as log_file:
        log_file.write("true\n" if dog_detected else "false\n")

    display.RenderOnce(img, width, height)
    display.SetTitle("Dog Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
