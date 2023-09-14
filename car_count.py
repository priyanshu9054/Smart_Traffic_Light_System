from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import*
cap = cv2.VideoCapture('video.mp4')
#cap = cv2.VideoCapture(0)
#cap.set(3,1280)
#cap.set(4, 720)

model=YOLO("../YOLO-weights/yolov8n.pt")

classNames= ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
             "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "handbag", "tie", "suitcase", "frisbee", 
             "skis", "snowboard", "sports ball", "kite", "basebell bat", "baseball glove", "skateboard", "snurfboard", 
             "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwhich",
             "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
             "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
             "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "tooth brush"
             ] 

#Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


limits = [0,400,1400,400]
totalCount=[]

while True:
    success, img = cap.read()
    
    results=model(img, stream=True)
    
    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:

            #Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
            #print(x1,y1,x2,y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,25),3)
            w, h = x2-x1,y2-y1
            #cvzone.cornerRect(img,(x1,y1,w,h),l=9)
            #Confidence
            conf=math.ceil((box.conf[0]*100))/100
            print(conf)
            
            #Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]


            if currentClass == "car" or currentClass == "truck" or currentClass =="motorbike" or currentClass == "bus" and conf>0.3:
                #cvzone.putTextRect(img, f'{currentClass}{conf}',(max(0, x1), max(35, y1)),
                 #                  scale=0.6, thickness=1,offset=3)
                cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=5)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

                
    resultsTracker = tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)

    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
        print(result)
        cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=2, colorR=(255,0,255))
        cvzone.putTextRect(img, f'{int(id)}',(max(0, x1), max(35, y1)),
                                   scale=2, thickness=6,offset=10)
        
        cx,cy = x1+w/2,y1+h/2
        #cv2.circle(img,(cx,cy),1,(255,0,255),cv2.FILLED)
       
        if limits[0]<cx< limits[2] and limits[1]-20<cy<limits[1]+20:
            if totalCount.count(id) ==0:
               totalCount.append(id)


    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50,50))

    cv2.imshow("Image",img)
    
    cv2.waitKey(1)