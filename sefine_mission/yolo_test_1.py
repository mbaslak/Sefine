import cv2
import numpy as np

# 1. Hazır ONNX modelini yüklüyoruz
model_yolu = "yolov8n.onnx"
net = cv2.dnn.readNetFromONNX(model_yolu)

# 2. JETSON İÇİN EN KRİTİK KISIM: İşlemleri Jetson GPU'suna aktarıyoruz
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Örnek bir resim okuyalım (veya kameradan bir kare)
frame = cv2.imread("test.jpg")
h, w, _ = frame.shape

# 3. Görüntüyü modelin beklediği formata (Blob) getiriyoruz
# (Örnek: 640x640 boyut, RGB renk düzeni, 0-1 arası ölçekleme)
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)

# 4. Görüntüyü modele verip tahmini alıyoruz
net.setInput(blob)
outputs = net.forward()



print("Model tahmini başarıyla alındı! Çıktı boyutu:", outputs.shape)