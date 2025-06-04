import torch
import torch.nn as nn
import torch.nn.functional as F

class AgeCNN(nn.Module):  # Age tahmini için Konvolüsyonel Sinir Ağı sınıfı
    def __init__(self):
        super(AgeCNN, self).__init__()
       
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)   # 1. konvolüsyon katmanı: 3 renk kanalından (RGB) 32 filtre çıkarır, filtre boyutu 3x3
      
        self.pool = nn.MaxPool2d(2, 2)      # Max pooling katmanı: 2x2 boyutunda havuzlama yapar, boyutları yarıya indirir . Havuzlma boyutları düşürür ve önemli noktaları çıkartır
 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)        # 2. konvolüsyon katmanı: 32 kanaldan 64 filtre çıkarır, filtre boyutu 3x3

        self.fc1 = nn.Linear(64 * 30 * 30, 128)     # Tam bağlantılı (fully connected) katman: 64 kanal, 30x30 boyutundaki çıkışı 128 boyutlu vektöre dönüştürür     
        self.fc2 = nn.Linear(128, 1)# Çıkış katmanı: 128 boyutlu vektörü 1 boyuta indirir (yaş tahmini için regresyon çıktısı)


    def forward(self, x):
        # 1. konvolüsyon -> ReLU aktivasyonu -> Max pooling
        x = self.pool(F.relu(self.conv1(x)))  # Çıktı boyutu: (batch_size, 32, 63, 63)
        # 2. konvolüsyon -> ReLU aktivasyonu -> Max pooling
        x = self.pool(F.relu(self.conv2(x)))  # Çıktı boyutu: (batch_size, 64, 30, 30)
        # Çok boyutlu tensörü tek boyutlu hale getir (flatten)
        x = x.view(-1, 64 * 30 * 30)
        # İlk tam bağlantılı katman -> ReLU aktivasyonu
        x = F.relu(self.fc1(x))
        # Son katman: yaş tahmini için tek değerli çıktı
        x = self.fc2(x)
        return x
