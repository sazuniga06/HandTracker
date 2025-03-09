# Hand Tracker

## 📌 Descripción General
Hand Tracker es un proyecto basado en **OpenCV** y **Mediapipe** que permite detectar y rastrear manos en tiempo real a través de una cámara web. Utiliza el modelo de detección de manos de **Mediapipe** para identificar puntos clave de la mano y visualizarlos con conexiones y marcas en la imagen.

## 🚀 Características
- Detección y seguimiento de hasta **2 manos** simultáneamente.
- Identificación de **21 puntos clave** por mano.
- Cálculo de posiciones de los puntos para futuras aplicaciones como gestos y controles.
- Visualización en tiempo real con FPS mostrados en pantalla.

## 🛠 Tecnologías Utilizadas
- **Python 3**
- **OpenCV** para el procesamiento de imagen.
- **Mediapipe** para la detección y rastreo de manos.
- **NumPy** (opcional) para operaciones matemáticas adicionales.

## 📦 Instalación

1. Instala las dependencias:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

## 🎯 Uso
1. Asegúrate de que tu cámara esté conectada.
2. Ejecuta el script y mueve las manos frente a la cámara.
3. Verás los puntos de referencia de la mano dibujados en la pantalla.
4. Presiona **ESC** para salir.

## 📌 Próximos Pasos
- Implementación de reconocimiento de gestos.
- Integración con aplicaciones interactivas.
- Optimizar el rendimiento en dispositivos con baja capacidad.

## 📜 Licencia
Este proyecto está bajo la licencia **MIT**. Puedes usarlo y modificarlo libremente.

---
💡 **Contribuciones y sugerencias son bienvenidas!** 😊

