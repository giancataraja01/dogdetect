import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")
display = jetson.utils.glDisplay()
font = jetson.utils.cudaFont()

with open("detection_logs.txt", "w") as log_file:
    log_file.write("false\n")

while display.IsOpen():
    img, width, height = camera.CaptureRGBA()
    detections = net.Detect(img, width, height, overlay="none")

    dog_detected = False

    for detection in detections:
        class_desc = net.GetClassDesc(detection.ClassID).lower()
        if class_desc == "dog":
            dog_detected = True
            # Draw bounding box
            jetson.utils.cudaDrawRect(
                img,
                (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)),
                (255, 255, 0, 255)
            )
            # Draw label using cudaFont (no width argument)
            font.Overlay(
                "dog",
                int(detection.Left),
                max(0, int(detection.Top) - 20),
                (255, 255, 0, 255),
                img
            )

    with open("detection_logs.txt", "w") as log_file:
        log_file.write("true\n" if dog_detected else "false\n")

    display.RenderOnce(img, width, height)
    display.SetTitle("Dog Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
