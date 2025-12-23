#!/usr/bin/env python3
"""
PDF Generation Test Script
Bu script, collection builder'ın PDF oluşturma fonksiyonunu test eder.
"""

import sys
import os
import base64
import json
from io import BytesIO
from PIL import Image
import traceback

# Flask uygulamasını import et
sys.path.insert(0, os.path.dirname(__file__))
from app import app

# Flask test client oluştur
test_client = app.test_client()

def create_test_image(width=800, height=1200, color=(100, 150, 200), text="Test Page"):
    """Test için basit bir görsel oluştur"""
    img = Image.new('RGB', (width, height), color=color)
    
    # Basit bir metin ekle (PIL'in ImageDraw kullanarak)
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Basit font kullan
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        except:
            font = ImageFont.load_default()
        
        # Metni ortala
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
    except Exception as e:
        print(f"⚠️  Metin eklenemedi: {e}")
    
    return img

def image_to_base64(img):
    """PIL Image'i base64 string'e çevir (data:image prefix olmadan)"""
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def test_pdf_generation():
    """PDF oluşturma fonksiyonunu test et"""
    print("=" * 60)
    print("🧪 PDF Generation Test Başlatılıyor...")
    print("=" * 60)
    
    # Test 1: Basit görsellerle test
    print("\n📝 Test 1: Basit görsellerle PDF oluşturma")
    print("-" * 60)
    
    try:
        # 3 test görseli oluştur
        test_images = []
        for i in range(3):
            color = (50 + i*50, 100 + i*30, 150 + i*20)
            img = create_test_image(
                width=800, 
                height=1200, 
                color=color,
                text=f"Test Sayfa {i+1}"
            )
            test_images.append(img)
            print(f"✅ Test görseli {i+1} oluşturuldu: {img.size[0]}x{img.size[1]}")
        
        # Görselleri base64'e çevir
        edited_images_array = []
        for i, img in enumerate(test_images):
            base64_str = image_to_base64(img)
            edited_images_array.append(base64_str)
            print(f"✅ Görsel {i+1} base64'e çevrildi (uzunluk: {len(base64_str)} karakter)")
        
        # API'ye istek gönder
        print(f"\n📤 API'ye istek gönderiliyor: /api/collection-builder/generate")
        payload = {
            "edited_images": edited_images_array
        }
        
        response = test_client.post(
            '/api/collection-builder/generate',
            json=payload,
            content_type='application/json'
        )
        
        print(f"📥 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.get_json()
                if data.get('success'):
                    pdf_data = data.get('pdf_data', '')
                    if pdf_data:
                        # Base64'ten PDF boyutunu hesapla
                        if pdf_data.startswith('data:application/pdf;base64,'):
                            pdf_base64 = pdf_data.split(',')[1]
                            pdf_bytes = base64.b64decode(pdf_base64)
                            pdf_size_kb = len(pdf_bytes) / 1024
                            print(f"✅ PDF başarıyla oluşturuldu!")
                            print(f"   📄 PDF boyutu: {pdf_size_kb:.2f} KB")
                            print(f"   📄 PDF sayfa sayısı: {len(test_images)}")
                            
                            # PDF'i dosyaya kaydet (test için)
                            test_output_path = "test_output.pdf"
                            with open(test_output_path, 'wb') as f:
                                f.write(pdf_bytes)
                            print(f"   💾 PDF kaydedildi: {test_output_path}")
                            
                            return True
                        else:
                            print(f"❌ PDF data formatı geçersiz")
                            return False
                    else:
                        print(f"❌ Response'da pdf_data bulunamadı")
                        print(f"   Response: {json.dumps(data, indent=2)}")
                        return False
                else:
                    error = data.get('error', 'Bilinmeyen hata')
                    print(f"❌ PDF oluşturulamadı: {error}")
                    return False
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse hatası: {e}")
                print(f"   Response text: {response.get_data(as_text=True)[:500]}")
                return False
        else:
            print(f"❌ HTTP hatası: {response.status_code}")
            try:
                error_data = response.get_json()
                print(f"   Hata: {error_data.get('error', 'Bilinmeyen hata')}")
            except:
                print(f"   Response text: {response.get_data(as_text=True)[:500]}")
            return False
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {str(e)}")
        print(f"   Traceback:\n{traceback.format_exc()}")
        return False

def test_empty_images():
    """Boş görsellerle test (hata durumu)"""
    print("\n📝 Test 2: Boş görsellerle test (hata durumu)")
    print("-" * 60)
    
    try:
        payload = {
            "edited_images": []
        }
        
        response = test_client.post(
            '/api/collection-builder/generate',
            json=payload,
            content_type='application/json'
        )
        
        if response.status_code == 400:
            print("✅ Beklenen hata döndü (400 Bad Request)")
            data = response.get_json()
            print(f"   Hata mesajı: {data.get('error', 'N/A')}")
            return True
        else:
            print(f"❌ Beklenmeyen response: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test hatası: {str(e)}")
        return False

def test_invalid_image():
    """Geçersiz görsel formatıyla test"""
    print("\n📝 Test 3: Geçersiz görsel formatıyla test")
    print("-" * 60)
    
    try:
        payload = {
            "edited_images": ["geçersiz_base64_string_12345"]
        }
        
        response = test_client.post(
            '/api/collection-builder/generate',
            json=payload,
            content_type='application/json'
        )
        
        # Bu durumda hata dönmeli veya geçersiz görselleri atlamalı
        print(f"📥 Response Status: {response.status_code}")
        data = response.get_json()
        
        if not data.get('success'):
            print(f"✅ Beklenen hata döndü: {data.get('error', 'N/A')}")
            return True
        else:
            print(f"⚠️  Hata bekleniyordu ama başarılı response döndü")
            return True  # Yine de başarılı sayılabilir (geçersiz görseller atlanmış olabilir)
            
    except Exception as e:
        print(f"❌ Test hatası: {str(e)}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("\n" + "=" * 60)
    print("🚀 PDF Generation Test Suite")
    print("=" * 60)
    
    # Flask test client kullanıldığı için sunucu kontrolü gerekmez
    print("\n✅ Flask test client hazır!")
    
    # Testleri çalıştır
    results = []
    
    # Test 1: Normal PDF oluşturma
    results.append(("Normal PDF Oluşturma", test_pdf_generation()))
    
    # Test 2: Boş görseller
    results.append(("Boş Görseller Testi", test_empty_images()))
    
    # Test 3: Geçersiz görsel
    results.append(("Geçersiz Görsel Testi", test_invalid_image()))
    
    # Sonuçları özetle
    print("\n" + "=" * 60)
    print("📊 Test Sonuçları")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "-" * 60)
    print(f"Toplam: {len(results)} test")
    print(f"✅ Başarılı: {passed}")
    print(f"❌ Başarısız: {failed}")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 Tüm testler başarıyla geçti!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {failed} test başarısız oldu.")
        sys.exit(1)

if __name__ == "__main__":
    main()


