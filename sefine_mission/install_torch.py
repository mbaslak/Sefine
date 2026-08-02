import urllib.request
import re
import os
import ssl

# SSL hatalarını bypass etmek için
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# NVIDIA'nın JetPack 6 sürümleri için olan depoları
urls = [
    "https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/",
    "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/",
    "https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/"
]

basarili = False

for url in urls:
    print(f"Taranıyor: {url}")
    try:
        req = urllib.request.urlopen(url, context=ctx)
        html = req.read().decode('utf-8')
        links = re.findall(r'href=[\'\"]?([^\'\" >]+)', html)
        
        # Python 3.10 (cp310) ve Jetson (aarch64) uyumlu dosyayı filtrele
        wheels = [l for l in links if 'torch-' in l and 'cp310' in l and 'aarch64' in l and l.endswith('.whl')]
        
        if wheels:
            dosya_linki = url + wheels[-1]
            print(f"\n[BULUNDU] Dosya indiriliyor:\n{dosya_linki}")
            print("(Bu işlem yaklaşık 800MB indirecektir, lütfen internet hızınıza göre bekleyin...)\n")
            
            # wget ile dosyayı indir (terminalde ilerleme çubuğu göstererek)
            os.system(f"wget -q --show-progress -O torch_jetson.whl {dosya_linki}")
            
            print("\nİndirme tamamlandı! Sisteme zorla kuruluyor...")
            os.system("pip install --force-reinstall torch_jetson.whl")
            
            basarili = True
            break
    except Exception as e:
        continue

if not basarili:
    print("\nSunucuda uygun dosya bulunamadı veya bağlantı reddedildi.")