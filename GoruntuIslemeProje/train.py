import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model import AgeCNN  # Daha önce tanımladığın model sınıfını alır
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class AgeDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []  # (resim_yolu, yaş) tuple'larını tutacak liste
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resimleri 128x128 boyutuna getir
            transforms.ToTensor(),  # Resimleri tensöre dönüştür
        ])
        for age_dir in os.listdir(root_dir):  # Klasör içinde yaş etiketleri (klasör isimleri) dolaşılır
            age_path = os.path.join(root_dir, age_dir)
            if os.path.isdir(age_path):  # Eğer bu bir klasörse (yaş etiketi klasörü)
                for fname in os.listdir(age_path):  # Klasördeki tüm dosyaları (görselleri) dolaş
                    # Görselin tam yolu ve yaş etiketini listeye ekle (float'a çevir)
                    self.samples.append((os.path.join(age_path, fname), float(age_dir)))

    def __len__(self):
        return len(self.samples)  # Veri kümesindeki toplam örnek sayısı

    def __getitem__(self, idx):
        img_path, age = self.samples[idx]  # İstenen indeksdeki veri ve yaş etiketi
        image = Image.open(img_path).convert('RGB')  # Görseli aç ve RGB'ye dönüştür
        # Görseli transform et ve yaş bilgisini float tensor olarak döndür
        return self.transform(image), torch.tensor([age], dtype=torch.float32)



def train():
    print("Training started...")  # Eğitim başlangıç 
    dataset = AgeDataset("veri")  # Veri klasörünü yükle
    
    from torch.utils.data import random_split
    total_size = len(dataset)  # Toplam veri sayısı
    val_size = int(0.2 * total_size)  # %20 doğrulama (validation) seti olarak ayır
    train_size = total_size - val_size  # Kalan %80 eğitim seti
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  # Veri setini böl
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)  # Eğitim için dataloader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)  # Doğrulama için dataloader
    
    model = AgeCNN()  # Modeli oluştur
    criterion = nn.MSELoss()  # Kayıp fonksiyonu: Ortalama Kare Hata (regresyon için uygun)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizasyon algoritması Adam, öğrenme hızı 0.001
    
    train_losses = []  # Eğitim kayıplarını tutacak liste
    val_mse_losses = []  # Doğrulama MSE kayıplarını tutacak liste
    val_mae_losses = []  # Doğrulama MAE (ortalama mutlak hata) kayıplarını tutacak liste
    
    for epoch in range(10):  # 10 epoch boyunca eğitim yap
        model.train()  # Modeli eğitim moduna al
        running_loss = 0.0  # O anki epoch'un toplam kaybı
        
        for inputs, labels in train_loader:  # Eğitim verisi üzerinden mini-batch'ler ile geç
            optimizer.zero_grad()  # Önceki adımın gradyanlarını sıfırla
            outputs = model(inputs)  # Model çıktısını hesapla
            loss = criterion(outputs, labels)  # Kayıp hesapla
            loss.backward()  # Gradyanları hesapla
            optimizer.step()  # Ağırlıkları güncelle
            running_loss += loss.item()  # Kayıpları topla
        
        # Doğrulama aşaması
        model.eval()  # Modeli doğrulama moduna al (gradyan hesaplama kapalı)
        val_mse = 0.0
        val_mae = 0.0
        with torch.no_grad():  # Gradyan hesaplama kapalı
            for inputs, labels in val_loader:  # Doğrulama verisi üzerinde geç
                outputs = model(inputs)  # Tahminleri al
                val_mse += nn.MSELoss()(outputs, labels).item()  # MSE hesapla ve topla
                val_mae += nn.L1Loss()(outputs, labels).item()  # MAE hesapla ve topla
        
        # Ortalama değerleri hesapla
        avg_train_loss = running_loss / len(train_loader)
        avg_val_mse = val_mse / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        
        # Kayıpları listeye ekle
        train_losses.append(avg_train_loss)
        val_mse_losses.append(avg_val_mse)
        val_mae_losses.append(avg_val_mae)
        
        # Her epoch sonunda sonuçları yazdır
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val MSE={avg_val_mse:.4f}, Val MAE={avg_val_mae:.4f}")
    
    torch.save(model.state_dict(), "yas_model.pth")  # Eğitilmiş modeli kaydet
    print("Model kaydedildi: yas_model.pth")
    
    # Eğitim ve doğrulama kayıplarını grafik olarak çizdir
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)  # İlk grafik alanı
    plt.plot(train_losses, label="Train Loss (MSE)")  # Eğitim kaybı grafiği
    plt.plot(val_mse_losses, label="Validation Loss (MSE)")  # Doğrulama kaybı grafiği
    plt.title("MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)  # İkinci grafik alanı
    plt.plot(val_mae_losses, label="Validation MAE", color='orange')  # Doğrulama MAE grafiği
    plt.title("MAE (Yaş Farkı)")
    plt.xlabel("Epoch")
    plt.ylabel("Ortalama Hata")
    plt.legend()
    
    plt.tight_layout()  # Grafiklerin düzgün yerleşimi
    plt.show()  # Grafik gösterimi

if __name__ == "__main__":
    train()  # Dosya direkt çalıştırıldığında train fonksiyonunu çağır
