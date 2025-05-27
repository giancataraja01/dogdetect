import jetson.inference
import jetson.utils

# Load the object detection model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")
display = jetson.utils.glDisplay()

# Initialize detection log file
with open("detection_logs.txt", "w") as log_file:
    log_file.write("false\n")

while display.IsOpen():
    img, width, height = camera.CaptureRGBA()
    detections = net.Detect(img, width, height)

    dog_detected = False

    for detection in detections:
        class_name = net.GetClassDesc(detection.ClassID).lower()
        if class_name == "dog":
            dog_detected = True
            # Draw bounding box only for dogs (green rectangle)
            jetson.utils.cudaDrawRect(img,
                                     (int(detection.Left),
                                      int(detection.Top),
                                      int(detection.Right - detection.Left),
                                      int(detection.Bottom - detection.Top)),
                                     (0, 255, 0, 255))

    # Update detection log file
    with open("detection_logs.txt", "w") as log_file:
        log_file.write("true\n" if dog_detected else "false\n")

    # Render the image with bounding boxes (only dogs)
    display.RenderOnce(img, width, height)
    display.SetTitle(f"Dog Detection | Network {net.GetNetworkFPS():.0f} FPS")
