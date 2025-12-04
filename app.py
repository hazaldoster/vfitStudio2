from flask import Flask, render_template, request, jsonify
from PIL import Image
from io import BytesIO
import base64
import io
import logging
import traceback
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Logging ayarı
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Gemini API Key - .env dosyasından al
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL_ID = os.environ.get('GEMINI_MODEL_ID', 'gemini-3-pro-image-preview')


@app.route('/vfit')
def vfit():
    logger.info("VFit Studio sayfası görüntüleniyor")
    return render_template('vfit.html')


@app.route('/api/vfit-tryon', methods=['POST'])
def vfit_tryon():
    """
    Virtual Try-On API endpoint - Gemini ile kıyafet deneme
    Model görseli + Ürün görseli alır, Gemini'a gönderir ve sonuç döndürür
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'JSON verisi gerekli'}), 400
        
        model_image = data.get('model_image')
        product_image = data.get('product_image')
        garment_type = data.get('garment_type', 'upper')
        fit_style = data.get('fit_style', 'natural')
        additional_instructions = data.get('additional_instructions', '')
        
        if not model_image:
            return jsonify({'success': False, 'error': 'Model görseli gerekli'}), 400
        
        # Ürün görseli yoksa ek talimat zorunlu
        if not product_image and not additional_instructions:
            return jsonify({'success': False, 'error': 'Ürün görseli yüklenmediyse ek talimat girmelisiniz'}), 400
        
        logger.info(f"🎽 VFit Try-On başlatılıyor - Kıyafet türü: {garment_type}, Fit: {fit_style}")
        logger.info(f"📷 Ürün görseli: {'Var' if product_image else 'Yok'}")
        
        # Garment type Turkish mapping
        garment_types = {
            'upper': 'üst giyim (tişört, gömlek, bluz, kazak)',
            'lower': 'alt giyim (pantolon, etek, şort)',
            'dress': 'elbise',
            'outerwear': 'dış giyim (ceket, mont, kaban)',
            'accessories': 'aksesuar (şal, kravat, şapka)'
        }
        
        fit_styles = {
            'natural': 'doğal ve rahat oturan',
            'slim': 'vücuda oturan slim fit',
            'loose': 'serbest ve rahat',
            'oversized': 'oversize, bol kesim'
        }
        
        garment_desc = garment_types.get(garment_type, 'kıyafet')
        fit_desc = fit_styles.get(fit_style, 'doğal')
        
        # Process images
        def process_base64_image(base64_str):
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
            return base64.b64decode(base64_str)
        
        model_image_data = process_base64_image(model_image)
        model_pil = Image.open(io.BytesIO(model_image_data))
        logger.info(f"📸 Model görsel boyutu: {model_pil.size}")
        
        # Ürün görseli varsa işle
        product_pil = None
        if product_image:
            product_image_data = process_base64_image(product_image)
            product_pil = Image.open(io.BytesIO(product_image_data))
            logger.info(f"👕 Ürün görsel boyutu: {product_pil.size}")
        
        # Build the prompt for Gemini based on whether product image exists
        if product_image:
            # Ürün görseli var - standart try-on prompt
            base_prompt = f"""Sen profesyonel bir virtual try-on uzmanısın.

Sana iki görsel veriyorum:
1. İLK GÖRSEL (MODEL): Bir model/manken fotoğrafı
2. İKİNCİ GÖRSEL (ÜRÜN): Bir {garment_desc} ürünü

GÖREV: İkinci görseldeki kıyafeti, birinci görseldeki modelin üzerine giydirerek yeni bir görsel oluştur.

KRİTİK KURALLAR - MUTLAKA UYULMALI:
1. MODEL GÖRSELİ BİREBİR AYNI KALMALI:
   - Modelin POZU kesinlikle değişmemeli
   - Modelin ÇEKİM AÇISI (kamera açısı) kesinlikle değişmemeli
   - Modelin DURUŞU, EL-KOL POZİSYONU aynı kalmalı
   - Modelin YÜZÜ, SAÇI, CİLT RENGİ değişmemeli
   - ARKA PLAN tamamen aynı kalmalı
   - IŞIKLANDIRMA ve GÖLGELER aynı kalmalı
   - GÖRSEL BOYUTU ve ORANI aynı kalmalı

2. SADECE KIYAFET DEĞİŞMELİ:
   - Ürün görselindeki kıyafet modelin üzerine giydirilmeli
   - Kıyafet {fit_desc} şekilde modele uymalı
   - Kıyafetin rengi, deseni, dokusu BİREBİR korunmalı
   - Kıyafet modelin vücut hatlarına doğal şekilde oturmalı

ÖZET: Input model görseli OUTPUT'ta BİREBİR aynı olmalı, tek fark üzerindeki kıyafet olmalı."""
        else:
            # Ürün görseli yok - sadece talimatlarla çalış (TAM ÖZGÜRLÜK)
            base_prompt = f"""Bu görseli düzenle ve yeni bir görsel oluştur.

SENİN GÖREVİN: Aşağıdaki talimatları AYNEN uygula. Talimatlar ne diyorsa onu yap.

TALİMATLAR:
{additional_instructions}

EK BİLGİLER:
- İstenen kıyafet türü: {garment_desc}
- İstenen oturma stili: {fit_desc}

UYGULAMA:
- Talimatlarda ne isteniyorsa SADECE onu yap
- Kısıtlama yok, talimatları takip et
- Gerçekçi ve profesyonel sonuç üret
- Değişiklikler doğal görünmeli

ŞİMDİ bu talimatlara göre yeni görsel oluştur."""

        if product_image and additional_instructions:
            base_prompt += f"\n\nEK TALİMATLAR (bunları da uygula): {additional_instructions}"
        
        if product_image:
            base_prompt += "\n\nLütfen bu virtual try-on görselini oluştur."
        
        # Call Gemini API
        if not GEMINI_API_KEY:
            raise Exception("Google AI API key yapılandırılmamış")
        
        import google.generativeai as genai_client
        
        genai_client.configure(api_key=GEMINI_API_KEY)
        
        model = genai_client.GenerativeModel(GEMINI_MODEL_ID)
        
        # Prepare content: prompt + images
        if product_pil:
            contents = [base_prompt, model_pil, product_pil]
        else:
            contents = [base_prompt, model_pil]
        
        logger.info(f"🚀 Gemini API'ye gönderiliyor ({GEMINI_MODEL_ID})...")
        logger.info(f"📝 Prompt: {base_prompt[:200]}...")
        
        response = model.generate_content(contents)
        
        # Extract generated image from response
        if response.candidates and response.candidates[0].content.parts:
            image_parts = [
                part.inline_data.data
                for part in response.candidates[0].content.parts
                if hasattr(part, 'inline_data') and part.inline_data
            ]
            
            if image_parts:
                # Process the generated image
                image_data = image_parts[0]
                
                try:
                    generated_image = Image.open(BytesIO(image_data))
                except Exception:
                    try:
                        decoded_data = base64.b64decode(image_data)
                        generated_image = Image.open(BytesIO(decoded_data))
                    except Exception as e:
                        raise Exception(f"Görsel işlenemedi: {str(e)}")
                
                # Convert to base64 for frontend
                img_buffer = BytesIO()
                generated_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                result_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                
                logger.info("✅ VFit Try-On başarıyla tamamlandı")
                
                return jsonify({
                    'success': True,
                    'result_image': f"data:image/png;base64,{result_base64}",
                    'garment_type': garment_type,
                    'fit_style': fit_style
                })
            else:
                # No image in response, check for text response
                text_response = response.text if hasattr(response, 'text') else str(response)
                logger.warning(f"⚠️ Gemini görsel döndürmedi. Yanıt: {text_response[:200]}")
                raise Exception("AI modeli görsel üretemedi. Lütfen farklı görseller deneyin.")
        else:
            raise Exception("AI modelinden yanıt alınamadı")
            
    except Exception as e:
        logger.error(f"❌ VFit Try-On hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/chat_edit_image', methods=['POST'])
def chat_edit_image():
    """
    Görsel düzenleme endpoint'i - Edit modal için
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'error': 'JSON verisi gerekli'}), 400
        
        prompt = data.get('prompt', '')
        session_id = data.get('session_id', '')
        uploaded_images = data.get('uploaded_images', [])
        
        if not prompt:
            return jsonify({'status': 'error', 'error': 'Prompt gerekli'}), 400
        
        if not uploaded_images:
            return jsonify({'status': 'error', 'error': 'Görsel gerekli'}), 400
        
        logger.info(f"🖼️ Görsel düzenleme başlatılıyor - Session: {session_id}")
        logger.info(f"📝 Prompt: {prompt}")
        
        # Get the input image
        input_image_data = None
        for img in uploaded_images:
            if img.get('type') == 'input':
                input_image_data = img.get('dataUrl')
                break
        
        if not input_image_data:
            return jsonify({'status': 'error', 'error': 'Input görsel bulunamadı'}), 400
        
        # Process base64 image
        def process_base64_image(base64_str):
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
            return base64.b64decode(base64_str)
        
        image_bytes = process_base64_image(input_image_data)
        input_pil = Image.open(io.BytesIO(image_bytes))
        
        logger.info(f"📸 Input görsel boyutu: {input_pil.size}")
        
        # Call Gemini API
        if not GEMINI_API_KEY:
            raise Exception("Google AI API key yapılandırılmamış")
        
        import google.generativeai as genai_client
        genai_client.configure(api_key=GEMINI_API_KEY)
        
        model = genai_client.GenerativeModel(GEMINI_MODEL_ID)
        
        # Build edit prompt
        edit_prompt = f"""Bu görseli düzenle. 
        
Düzenleme talimatı: {prompt}

Önemli:
- Görselin genel yapısını koru
- Sadece istenen değişiklikleri yap
- Yüksek kaliteli sonuç üret
- Doğal ve profesyonel görünüm sağla"""
        
        contents = [
            edit_prompt,
            input_pil
        ]
        
        logger.info(f"🚀 Gemini API'ye gönderiliyor ({GEMINI_MODEL_ID})...")
        
        response = model.generate_content(contents)
        
        # Extract generated image from response
        if response.candidates and response.candidates[0].content.parts:
            image_parts = [
                part.inline_data.data
                for part in response.candidates[0].content.parts
                if hasattr(part, 'inline_data') and part.inline_data
            ]
            
            if image_parts:
                image_data = image_parts[0]
                
                try:
                    generated_image = Image.open(BytesIO(image_data))
                except Exception:
                    try:
                        decoded_data = base64.b64decode(image_data)
                        generated_image = Image.open(BytesIO(decoded_data))
                    except Exception as e:
                        raise Exception(f"Görsel işlenemedi: {str(e)}")
                
                # Convert to base64 for frontend
                img_buffer = BytesIO()
                generated_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                result_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                
                logger.info("✅ Görsel düzenleme başarıyla tamamlandı")
                
                return jsonify({
                    'status': 'success',
                    'generated_images': [f"data:image/png;base64,{result_base64}"]
                })
            else:
                text_response = response.text if hasattr(response, 'text') else str(response)
                logger.warning(f"⚠️ Gemini görsel döndürmedi. Yanıt: {text_response[:200]}")
                raise Exception("AI modeli görsel üretemedi.")
        else:
            raise Exception("AI modelinden yanıt alınamadı")
            
    except Exception as e:
        logger.error(f"❌ Görsel düzenleme hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


# Ana sayfa yönlendirmesi
@app.route('/')
def index():
    return vfit()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

