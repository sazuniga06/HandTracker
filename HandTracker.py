import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

ptime = 0
ctime = 0

while True:
    success, img = cap.read()
    
    if not success:
        print("Ignoring empty camera frame.")
        continue    
    
    img=cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)  
    #print(results.multi_hand_landmarks)  
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                #if id == 0:
                    #cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
                cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
                    
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)            
    
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    