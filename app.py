import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import json

def run_script(script_path, output_file):
    subprocess.run(['python', script_path, output_file])

def load_angles(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def main():
    # İlk ve ikinci scriptin yolları
    script1_path = 'MediaPipePoseEstimation.py'  # İlk kod dosyası (SIFT ve OpenCV)
    script2_path = 'app2.py'  # İkinci kod dosyası (MediaPipe)

    # Açılar için geçici dosyalar
    angles1_file = 'angles1.json'
    angles2_file = 'angles2.json'

    # Scriptleri sırayla çalıştır
    print("İlk script çalıştırılıyor...")
    run_script(script1_path, angles1_file)

    print("İkinci script çalıştırılıyor...")
    run_script(script2_path, angles2_file)

    # Açılar yükleniyor
    angles1 = load_angles(angles1_file)
    angles2 = load_angles(angles2_file)

    # Veriyi DataFrame'e dönüştür
    max_length = max(len(angles1), len(angles2))
    angles1 += [None] * (max_length - len(angles1))
    angles2 += [None] * (max_length - len(angles2))

    df = pd.DataFrame({
        'Frame': range(1, max_length + 1),
        'Script1_Aci': angles1,
        'Script2_Aci': angles2
    })

    # Tabloyu kaydet
    df.to_csv('aci_karsilastirma.csv', index=False)
    print("Açı karşılaştırma tablosu oluşturuldu: aci_karsilastirma.csv")

    # Grafiği çizdir
    plt.figure(figsize=(10, 6))
    plt.plot(df['Frame'], df['Script1_Aci'], label='M Açısı', linewidth=2)
    plt.plot(df['Frame'], df['Script2_Aci'], label='Script  Açısı', linewidth=2, linestyle='--')
    plt.xlabel('Frame')
    plt.ylabel('Açı (Derece)')
    plt.title('Scriptler Arası Açı Karşılaştırması')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
