import cv2
import mediapipe as mp
import time

class HandTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None  # Para almacenar los resultados de la detección

    def findHands(self, img, draw=True):
        if img is None:
            print("⚠ Advertencia: imagen no capturada correctamente.")
            return None  # Evita procesar si la imagen está vacía

        img = cv2.flip(img, 1)  # Voltear la imagen para un efecto espejo
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # Guardar los resultados

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def getHandPositions(self, img, drawing=True):
        """ Devuelve una lista con las coordenadas (x, y) de los puntos de referencia de todas las manos detectadas. """
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # Iterar sobre todas las manos detectadas
                hand_lm = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    hand_lm.append((id, cx, cy))
                    if drawing:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)  # Dibujar puntos
                lmList.append(hand_lm)  # Agregar lista de puntos de cada mano

        return lmList

def main():
    ptime = 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara.")
        return

    detector = HandTracker()

    while True:
        success, img = cap.read()
        if not success or img is None:
            print("⚠ Error: No se pudo capturar la imagen de la cámara.")
            continue  # Saltar esta iteración si la imagen es inválida

        img = detector.findHands(img)
        if img is None:
            continue  # Evita mostrar una imagen inválida

        lmlist = detector.getHandPositions(img)  # Obtener lista de puntos de las manos

        if len(lmlist) != 0:
            print(f"Manos detectadas: {len(lmlist)}")
            print(f"Primer punto de la primera mano: {lmlist[0][0]}")  # Coordenadas del primer punto de la primera mano

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == 27:  # Presionar ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
