from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import*   
img = cv2.imread('day1.png')
img1 = cv2.imread('day2.png')
img2 = cv2.imread('day3.png')
img3 = cv2.imread('day4.png')
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

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


totalCount=[]

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



for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
        print(result)
        cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=2, colorR=(255,0,255))
        cvzone.putTextRect(img, f'{int(id)}',(max(0, x1), max(35, y1)),
                                   scale=2, thickness=6,offset=10)
        
        cx,cy = x1+w/2,y1+h/2
        #cv2.circle(img,(cx,cy),1,(255,0,255),cv2.FILLED)
       
        
        if totalCount.count(id) ==0:
               totalCount.append(id)
lane1 = (len(totalCount))

print('Total number of vehichles in lane1 =',lane1)

cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50,50))

cv2.imshow("Image",img)
    
cv2.waitKey(0)

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


totalCount=[]

results=model(img1, stream=True)
    
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
                cvzone.cornerRect(img1,(x1,y1,w,h),l=9,rt=5)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

               
resultsTracker = tracker.update(detections)



for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
        print(result)
        cvzone.cornerRect(img1,(x1,y1,w,h),l=9,rt=2, colorR=(255,0,255))
        cvzone.putTextRect(img1, f'{int(id)}',(max(0, x1), max(35, y1)),
                                   scale=2, thickness=6,offset=10)
        
        cx,cy = x1+w/2,y1+h/27
        #cv2.circle(img,(cx,cy),1,(255,0,255),cv2.FILLED)
       
        
        if totalCount.count(id) ==0:
               totalCount.append(id)
lane2 = (len(totalCount))

print('Total number of vehichles in lane2 =',lane2)

cvzone.putTextRect(img1, f' Count: {len(totalCount)}', (50,50))

cv2.imshow("Image",img1)
    
cv2.waitKey(0)

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


totalCount=[]

results=model(img2, stream=True)
    
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
                cvzone.cornerRect(img2,(x1,y1,w,h),l=9,rt=5)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

               
resultsTracker = tracker.update(detections)



for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
        print(result)
        cvzone.cornerRect(img2,(x1,y1,w,h),l=9,rt=2, colorR=(255,0,255))
        cvzone.putTextRect(img2, f'{int(id)}',(max(0, x1), max(35, y1)),
                                   scale=2, thickness=6,offset=10)
        
        cx,cy = x1+w/2,y1+h/2
        #cv2.circle(img,(cx,cy),1,(255,0,255),cv2.FILLED)
       
        
        if totalCount.count(id) ==0:
               totalCount.append(id)
lane3 = (len(totalCount))

print('Total number of vehichles in lane3 =',lane3)

cvzone.putTextRect(img2, f' Count: {len(totalCount)}', (50,50))

cv2.imshow("Image",img2)
    
cv2.waitKey(0)

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


totalCount=[]

results=model(img3, stream=True)
    
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
                cvzone.cornerRect(img3,(x1,y1,w,h),l=9,rt=5)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

               
resultsTracker = tracker.update(detections)



for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
        print(result)
        cvzone.cornerRect(img3,(x1,y1,w,h),l=9,rt=2, colorR=(255,0,255))
        cvzone.putTextRect(img3, f'{int(id)}',(max(0, x1), max(35, y1)),
                                   scale=2, thickness=6,offset=10)
        
        cx,cy = x1+w/2,y1+h/2
        #cv2.circle(img,(cx,cy),1,(255,0,255),cv2.FILLED)
       
        
        if totalCount.count(id) ==0:
               totalCount.append(id)
lane4 = (len(totalCount))

print('Total number of vehichles in lane4 =',lane4)

cvzone.putTextRect(img3, f' Count: {len(totalCount)}', (50,50))

cv2.imshow("Image",img3)
    
cv2.waitKey(0)

if lane1 > lane2 and lane1 > lane3 and lane1 > lane4:
      largest = lane1
      print("LANE-1")
      print('turns yellow for 3 sec!')
      print('turns green for 10 sec!')
      print("turns yellow for 3 sec!")
      print("Back to red")
      print("-------------------------------")
      print("LANE-2")
      print("Remains red")
      print("-------------------------------")
      print("LANE-3")
      print("Remains red")
      print("-------------------------------")
      print("LANE-4")
      print("Remains red")
      print("-------------------------------")
elif lane2 > lane1 and lane2 > lane3 and lane2 > lane4:
      largest = lane2
      print("LANE-1")
      print("Remains red")
      print("-------------------------------")
      print("LANE-2")
      print('turns yellow for 3 sec!')
      print('turns green for 10 sec!')
      print("turns yellow for 3 sec!")
      print("Back to red")
      print("-------------------------------")
      print("LANE-3")
      print("Remains red")
      print("-------------------------------")
      print("LANE-4")
      print("Remains red")
      print("-------------------------------")
elif lane3 > lane1 and lane3 > lane2 and lane3 > lane4:
      largest = lane3
      print("LANE-1")
      print("Remains red")
      print("-------------------------------")
      print("LANE-2")
      print("Remains red")
      print("-------------------------------")
      print("LANE-3")
      print('turns yellow for 3 sec!')
      print('turns green for 10 sec!')
      print("turns yellow for 3 sec!")
      print("Back to red")
      print("-------------------------------")
      print("LANE-4")
      print("Remains red")
      print("-------------------------------")
else:
      largest = lane4
      print("LANE-1")
      print("Remains red")
      print("-------------------------------")
      print("LANE-2")
      print("Remains red")
      print("-------------------------------")
      print("LANE-3")
      print("Remains red")
      print("-------------------------------")
      print("LANE-4")
      print('turns yellow for 3 sec!')
      print('turns green for 10 sec!')
      print("turns yellow for 3 sec!")
      print("Back to red")
      print("-------------------------------")
      print("===============================")