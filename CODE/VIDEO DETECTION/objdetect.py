import cv2
import spacy 
import numpy as np 
from imutils import opencv2matplotlib
import matplotlib.pyplot as plt


# Load an NLP model
nlp = spacy.load("en_core_web_sm")
keywords="man with gun"

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Open a video source (you can also use a file path)
video_capture = cv2.VideoCapture(0)  # Replace 0 with the appropriate source

while True:
    # Capture each frame
    ret, frame = video_capture.read()
    height, width, channels = frame.shape

    # Use YOLO to detect objects in the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Process YOLO results
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    detected_objects = []

    # Compare YOLO-detected objects with keywords from the description
    for i in range(len(boxes)):
        if class_ids[i] in [0, 1, 2, 3, 4, 5]:  # Assuming classes 0-5 represent persons
            label = str(keywords)  # Assign description keywords to the detected object
            detected_objects.append(label)

    # Display the frame with bounding boxes
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, detected_objects[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame in a window
    cv2.imshow("Real-Time Object Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.namedWindow("Real-Time Object Detection", cv2.WINDOW_NORMAL)

plt.imshow(opencv2matplotlib(frame))
plt.title("Real-Time Object Detection")
plt.show()


# Release the video source and close all windows
video_capture.release()
cv2.destroyAllWindows()
