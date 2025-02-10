
# -*- coding: utf-8 -*-
import cv2
import torch
from ultralytics import YOLO

# iPhone Camera URL
stream_url = "http://192.168.1.3:4747/video"

# Open video stream
cap = cv2.VideoCapture(0)
model = YOLO('yolo11n_ncnn_model', task="detect")
model.conf = 0.6   # Güven eşiği
model.iou = 0.45   # IoU eşiği

# Video akışı açıkken işlemleri yap
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kamera bağlantısı başarısız!")
        break

    # YOLO modelini kullanarak nesne tespiti yap
    results = model(frame)

    # Sonuçları işle
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Nesnenin dikdörtgen koordinatları
            conf = box.conf[0].item()  # Güven skoru
            label = result.names[int(box.cls[0])]  # Nesne etiketi

            # Yalnızca belirli bir güven eşiğinin üzerindekileri göster
            if conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Çıktıyı göster
    cv2.imshow("YOLO Detection", frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera bağlantısını kapat
cap.release()
cv2.destroyAllWindows()
