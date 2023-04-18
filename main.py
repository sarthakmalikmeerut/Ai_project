from ultralytics import YOLO
import cv2
import smtplib
import cvzone
import math

vid = cv2.VideoCapture("C:\\Users\\sarth\\PycharmProjects\\yolo_project\\car_fire.mp4")
vid.set(3,1280)
vid.set(4,720)
model = YOLO("C:\\Users\\sarth\\PycharmProjects\\yolo_project\\best.pt")


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


while True:
    success,img = vid.read()
    results = model(img,stream = True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

            w,h = x2-x1 , y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))

            conf = math.ceil((box.conf[0]*100))/100
            if conf > 0.7 :
                s = smtplib.SMTP('smtp.gmail.com', 587)
                s.starttls()
                s.login("sarthakmalikmeerut@gmail.com", "doomgqtqjkqoosyc")
                message = "your location is jalandhar"
                s.sendmail("sarthakmalikmeerut@gmail.com", "sarthakmalikmeerut@gmail.com", message)
            cls = int(box.cls[0])

            cvzone.putTextRect(img,f'{names[cls]} {conf}', (max(0,x1), max(35,y1)))


    cv2.imshow("image",img)
    cv2.waitKey(1)
