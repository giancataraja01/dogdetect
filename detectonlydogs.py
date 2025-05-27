import jetson.inference
import jetson.utils

# Load the object detection model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")
display = jetson.utils.glDisplay()

# Initialize the log file with no detection initially
with open("detection_logs.txt", "w") as log_file:
    log_file.write("false\n")

while display.IsOpen():
    # Capture image from camera
    img, width, height = camera.CaptureRGBA()

    # Run detection without overlaying bounding boxes or labels
    detections = net.Process(img, width, height)

    dog_detected = False

    # Manually draw bounding boxes and labels only for dogs
    for detection in detections:
        class_desc = net.GetClassDesc(detection.ClassID).lower()
        if class_desc == "dog":
            dog_detected = True

            # Draw green bounding box around the dog
            jetson.utils.cudaDrawRect(img,
                                     int(detection.Left),
                                     int(detection.Top),
                                     int(detection.Right - detection.Left),
                                     int(detection.Bottom - detection.Top),
                                     (0, 255, 0, 255))  # RGBA green

            # Draw label text above the bounding box
            jetson.utils.cudaDrawText(img,
                                     class_desc,
                                     int(detection.Left),
                                     max(0, int(detection.Top) - 15),
                                     (0, 255, 0, 255), 2)  # RGBA green, size 2

    # Write detection result to log file
    with open("detection_logs.txt", "w") as log_file:
        log_file.write("true\n" if dog_detected else "false\n")

    # Render the image and update the window title with FPS
    display.RenderOnce(img, width, height)
    display.SetTitle("Dog Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
