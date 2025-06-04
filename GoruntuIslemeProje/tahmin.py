import cv2
import torch
from model import AgeCNN
from torchvision import transforms
from PIL import Image

model = AgeCNN()  # AgeCNN sınıfından bir model oluşturur.
model.load_state_dict(torch.load("yas_model.pth", map_location=torch.device("cpu"))) #yas_model.pth dosyasındaki eğitilmiş ağırlıkları yükler.
model.eval() #model.eval() ile tahmin (inference) moduna geçirir.

#Görüntü ön işleme 
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
]) #Yüz görsellerini modelin beklediği forma (128x128 boyut + tensör) dönüştürür

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")#OpenCV’nin ön tanımlı yüz tanıma modeliyle karedeki yüzleri bulur.

cap = cv2.VideoCapture(0)  # Kamerayı aç

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)) #PİL formatına çevir.(pillow)
        face_tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(face_tensor) # AGECNN SINIFINDAN OLUŞAN MODELİ İLE TAHMİNDE BULUN
            predicted_age = output.item() #YAŞ TAHMİNİ YAP

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{predicted_age:.1f} yas", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Kameradan Yas Tahmini", frame)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()
