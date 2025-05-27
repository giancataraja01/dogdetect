import jetson.inference
import jetson.utils

# Load the object detection model without automatic overlay
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

    # Clear any previous drawings by starting fresh on img
    # (The image buffer is refreshed on every capture)

    for detection in detections:
        class_name = net.GetClassDesc(detection.ClassID).lower()
        if class_name == "dog":
            dog_detected = True
            # Draw green bounding box for dog
            jetson.utils.cudaDrawRect(img,
                                     (int(detection.Left),
                                      int(detection.Top),
                                      int(detection.Right - detection.Left),
                                      int(detection.Bottom - detection.Top)),
                                     (0, 255, 0, 255))

            # Optional: Draw label text for dog
            label = f"{class_name} {detection.Confidence * 100:.1f}%"
            jetson.utils.cudaFont().OverlayText(img, 
                                               label,
                                               int(detection.Left),
                                               int(detection.Top) - 20,  # slightly above box
                                               (0, 255, 0, 255),  # green text
                                               size=20)

    # Update detection log file
    with open("detection_logs.txt", "w") as log_file:
        log_file.write("true\n" if dog_detected else "false\n")

    # Render only your manually drawn image
    display.RenderOnce(img, width, height)
    display.SetTitle(f"Dog Detection | Network {net.GetNetworkFPS():.0f} FPS")
