import jetson.inference
import jetson.utils

# Load the object detection model with threshold 0.5
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Open the camera (adjust device as needed)
camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")

# Create display window
display = jetson.utils.glDisplay()

# Initialize detection log file
with open("detection_logs.txt", "w") as log_file:
    log_file.write("false\n")

while display.IsOpen():
    # Capture image frame from the camera
    img, width, height = camera.CaptureRGBA()

    # Run detection WITHOUT auto overlay (Process() only)
    detections = net.Process(img, width, height)

    dog_detected = False

    # Manually draw bounding boxes and labels only for dogs
    for detection in detections:
        class_desc = net.GetClassDesc(detection.ClassID).lower()
        if class_desc == "dog":
            dog_detected = True

            # Draw bounding box (green color)
            jetson.utils.cudaDrawRect(img,
                                     int(detection.Left),
                                     int(detection.Top),
                                     int(detection.Right - detection.Left),
                                     int(detection.Bottom - detection.Top),
                                     (0, 255, 0, 255))

            # Draw label text above the bounding box
            jetson.utils.cudaDrawText(img,
                                     class_desc,
                                     int(detection.Left),
                                     max(0, int(detection.Top) - 15),
                                     (0, 255, 0, 255), 2)

    # Update the detection log file
    with open("detection_logs.txt", "w") as log_file:
        log_file.write("true\n" if dog_detected else "false\n")

    # Render the image with manual dog detections
    display.RenderOnce(img, width, height)

    # Update window title with current FPS
    display.SetTitle("Dog Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
