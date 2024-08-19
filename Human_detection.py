import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet(r"SAMPLE_PROJ\yolov3.weights", r"SAMPLE_PROJ\yolov3.cfg")
classes = []
with open(r"SAMPLE_PROJ\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load image
image = cv2.imread(r"C:\Users\sayli\OneDrive\Documents\SY_SEMIV\AIES\MINI_PROJ\SAMPLE_PROJ\img2.jpeg")
height, width, _ = image.shape

# Preprocess image
blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Run inference
output_layers_names = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers_names)

# Process detections
boxes = []
confidences = []
class_ids = []
for output in layer_outputs:
    for detection in output:
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
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-max suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes and labels
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        # color = colors[class_ids[i]]
        cv2.rectangle(image, (x, y), (x + w, y + h), (225,0,0), 2)
        cv2.putText(image, label, (x, y - 5), font, 1, (225,0,0), 2)

# Display output
cv2.imshow("YOLO Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()