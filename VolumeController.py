import cv2
import mediapipe as mp
import time
import numpy as np
import Hand1  # Importar el módulo Hand1
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER

# Configuración de la cámara
wcam, hcam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

ptime = 0  # Tiempo previo para calcular FPS
hand_tracker = Hand1.HandTracker(detectionCon=0.7)  # Instancia del rastreador de manos

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volumeR = volume.GetVolumeRange()
volumeMin = volumeR[0]
volumeMax = volumeR[1]

vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    
    if not success:
        print("Ignorando cuadro vacío.")
        continue    
    
    img = hand_tracker.findHands(img)
    lmList = hand_tracker.getHandPositions(img, drawing=False)
    
    if lmList and len(lmList[0]) > 8:  # Verificar que haya suficientes puntos antes de acceder a ellos
        x1, y1 = lmList[0][4][1], lmList[0][4][2]
        x2, y2 = lmList[0][8][1], lmList[0][8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)
        
        # Ajuste mejorado del volumen
        vol = np.interp(length, [30, 300], [volumeMin, volumeMax])
        volBar = np.interp(length, [30, 300], [400, 150])
        volPer = np.interp(length, [30, 300], [0, 100])
        
        # Redondeo del volumen para mejor control
        vol = round(vol, 2)
        volume.SetMasterVolumeLevel(vol, None)
        
        if length < 30:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
                
    # Cálculo de FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    
    # Mostrar FPS en pantalla
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Permite salir con la tecla 'esc'
        break

cap.release()
cv2.destroyAllWindows()
