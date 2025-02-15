import cv2
import numpy as np


net = cv2.dnn.readNet("SAMPLE_PROJ\yolov3.weights", "SAMPLE_PROJ\yolov3.cfg")
classes = []
with open("SAMPLE_PROJ\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


video = cv2.VideoCapture("SAMPLE_PROJ\VID.mp4") 

while True:
    ret, frame = video.read()
    if not ret:
        break

    height, width, _ = frame.shape

 
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

   
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

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

   
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

   
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 5), font, 1, (225, 0, 0), 2)

    cv2.imshow("YOLO Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
