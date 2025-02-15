import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# Sentetik veriler (Rastgele değerler atandı, gerçekte bunlar analizlerle belirlenir)
mahalleler = ["Cumhuriyet", "İstasyon", "Pınar"]
nufus_yogunlugu = np.array([70, 90, 80])  # Rastgele nüfus yoğunluğu
ulasim_alt_yapi = np.array([70, 90, 60])
maliyet = np.array([50, 80, 70])  # Maliyet (düşük olması avantajlıdır)
cevresel_etki = np.array([60, 80, 50])
sosyal_fayda = np.array([50, 70, 80])

maliyet_ters = 100 - maliyet

veriler = np.vstack([nufus_yogunlugu, ulasim_alt_yapi, maliyet_ters, cevresel_etki, sosyal_fayda])
kriter_agirliklari = softmax(veriler.sum(axis=1))

guzergah_skorlari = np.dot(kriter_agirliklari, veriler)

en_uygun_mahalle = mahalleler[np.argmax(guzergah_skorlari)]

print("Kriter Ağırlıkları:")
for k, v in zip(["Nüfus Yoğunluğu", "Ulaşım Altyapısı", "Maliyet", "Çevresel Etki", "Sosyal Fayda"], kriter_agirliklari):
    print(f"{k}: {v:.3f}")

print("\nGüzergah Skorları:")
for mahalle, skor in zip(mahalleler, guzergah_skorlari):
    print(f"{mahalle}: {skor:.2f}")

print(f"\nEn uygun güzergah: {en_uygun_mahalle}")
