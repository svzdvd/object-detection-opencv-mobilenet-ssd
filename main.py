import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

# img = cv2.imread('lena.png')

# Test: show image
# cv2.imshow('Output', img)
# cv2.waitKey(0)

# import class names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    if success:
        classIds, confidences, bboxes = net.detect(img, confThreshold=0.6)
        print(classIds, confidences, bboxes)

        if len(classIds) != 0:
            for classId, confidence, bbox in zip(classIds.flatten(), confidences.flatten(), bboxes):
                cv2.rectangle(img, bbox, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId - 1].upper(), (bbox[0] + 10, bbox[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

        cv2.imshow('Output', img)
        cv2.waitKey(1)