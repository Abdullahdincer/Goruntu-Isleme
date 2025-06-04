
import cv2
import os

# İşlenecek videolar: (video_adı, yaş)
videolarim = [
    ("Ben_21.mp4", 21),
    ("KadirAbi_46.mp4", 46),
    ("Mert_18.mp4", 18),
    ("Yusuf.mp4", 11),
    ("VeyselAbi_31.mp4", 31),
]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #görüntüdeki yüzleri tespit etmek için gerekli kütüphane
max_faces_per_video = 234  # Her video için maksimum yüz

for video_path, age_label in videolarim:
    save_dir = f"veri/{age_label}"
    os.makedirs(save_dir, exist_ok=True)

    # Eğer bu klasörde zaten 234 veya daha fazla veri varsa geç bir daha verilerin işlenmemesi için
    mevcut_kayitlar = len(os.listdir(save_dir))
    if mevcut_kayitlar >= max_faces_per_video:
        print(f"{video_path} -> Zaten işlenmiş ({mevcut_kayitlar} kayıt). Atlanıyor.")
        continue

    cap = cv2.VideoCapture(video_path)
    saved_count = mevcut_kayitlar  # Kaldığı yerden devam etme (istenirse)

    print(f"{video_path} -> İşleniyor...")

    while cap.isOpened() and saved_count < max_faces_per_video:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) #yüzler tespit edilir scalefactor ile görüntü ölçeklendirlidi

        for (x, y, w, h) in faces:
            if saved_count >= max_faces_per_video: #her görüntüden eşit bir şekilde al
                break

            if w < 50 or h < 50:
                continue  # Çok küçük yüzler alınmasın

            face = frame[y:y+h, x:x+w] #yüzü kırp
            face = cv2.resize(face, (128, 128)) #128*128 hale getir
            filename = os.path.join(save_dir, f"{age_label}_{saved_count}.jpg")
            cv2.imwrite(filename, face)
            print(f"{video_path} -> Kayıt edildi: {filename}")
            saved_count += 1

    cap.release()
    print(f"{video_path} -> Tamamlandı. Toplam kayıt: {saved_count}\n")

cv2.destroyAllWindows()
print("Tüm videolar işlendi. Her video için en fazla 234 yüz kaydedildi.")
