import cv2
from ultralytics import YOLO

# 1. Modeli yüklüyoruz (.pt veya Jetson için optimize ettiğimiz .engine)
model = YOLO("yolov8n.engine")

# 2. Kamera akışını başlatıyoruz (0: Varsayılan USB kamera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kamera akışı alınamadı!")
        break

    # 3. Modelden tahmin alıyoruz
    results = model(frame, verbose=False)

    # 4. Tespit edilen TÜM nesneleri döngüyle geziyoruz
    for box in results[0].boxes:
        
        # A) Kutu Koordinatları (x1, y1: Sol Üst Köşe | x2, y2: Sağ Alt Köşe)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # B) Doğruluk Skoru (Confidence Score - 0.0 ile 1.0 arası)
        confidence = float(box.conf[0])
        
        # C) Sınıf ID ve İsmi (Örn: 0 -> 'person', 2 -> 'car')
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # D) OTONOM SÜRÜŞ İÇİN KRİTİK: Nesnenin Tam Merkez Noktası (Center Point)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # -------------------------------------------------------------
        # OPENCV İLE GÖRÜNTÜ ÜZERİNE ÇİZİM YAPMA
        # -------------------------------------------------------------

        # 1. Dikdörtgen Çerçeve Çizme
        # cv2.rectangle(Görüntü, (x1, y1), (x2, y2), Renk(B,G,R), Kalınlık)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 2. Etiket Yazısı Hazırlama (Örn: "Araba %85")
        label = f"{class_name} %{confidence*100:.0f}"

        # 3. Yazıyı Kutu Üstüne Ekleme
        # cv2.putText(Görüntü, Metin, Pozisyon, Yazı Tipi, Ölçek, Renk, Kalınlık)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 4. Nesnenin Tam Ortasına Kırmızı Bir Nokta Koyma
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # -------------------------------------------------------------
        # BİLGİLERİ KULLANMA (Örn: ArduPilot / MAVSDK Yönlendirmesi)
        # -------------------------------------------------------------
        print(f"Tespit Edildi: {class_name} | Güven: %{confidence*100:.1f} | Merkez: X={center_x}, Y={center_y}")

    # İşlenmiş 'frame' artık üzerinde yeşil kutular ve yazılar olan karedir.
    # Monitörsüz sistemde bu 'frame'i bir .mp4 dosyasına yazabilir veya Flask ile web'e basabilirsiniz.

cap.release()