from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
from io import BytesIO
import base64
import io
import logging
import traceback
import os
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.utils import ImageReader
import tempfile

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


# Ana sayfa
@app.route('/')
def index():
    return render_template('homepage.html')


@app.route('/collection-builder')
def collection_builder():
    logger.info("Collection Builder sayfası görüntüleniyor")
    return render_template('collection_builder.html')


# PDF dosya yolları - glob ile bul
import glob

def find_pdf(pattern):
    """PDF dosyasını pattern ile bul"""
    base_dir = os.path.dirname(__file__)
    matches = glob.glob(os.path.join(base_dir, pattern))
    if matches:
        return matches[0]
    return None

LOOKBOOK_PDF = find_pdf("*LOOKBOOK*.pdf") or os.path.join(os.path.dirname(__file__), "MEHTAP ELAIDI FW '25 LOOKBOOK.pdf")
LINESHEET_PDF = find_pdf("*LINESHEET*.pdf") or os.path.join(os.path.dirname(__file__), "URBAN MUSE SS26 LINESHEET.pdf")


@app.route('/api/collection-builder/pages', methods=['GET'])
def get_pages():
    """PDF sayfalarını preview olarak döndür"""
    try:
        doc_type = request.args.get('type', 'lookbook')
        
        pdf_path = LOOKBOOK_PDF if doc_type == 'lookbook' else LINESHEET_PDF
        
        if not os.path.exists(pdf_path):
            return jsonify({'success': False, 'error': 'PDF dosyası bulunamadı'}), 404
        
        # PDF'i görsellere çevir
        images = convert_from_path(pdf_path, dpi=100)
        
        pages = []
        for i, img in enumerate(images):
            # Görseli base64'e çevir (preview için küçük boyut)
            img.thumbnail((300, 400), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            pages.append({
                'page_number': i + 1,
                'preview': f"data:image/png;base64,{img_base64}"
            })
        
        logger.info(f"✅ {len(pages)} sayfa yüklendi ({doc_type})")
        
        return jsonify({
            'success': True,
            'pages': pages,
            'total_pages': len(pages)
        })
        
    except Exception as e:
        logger.error(f"❌ Sayfa yükleme hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/collection-builder/generate-page', methods=['POST'])
def generate_page():
    """Tek bir sayfayı prompta göre düzenle"""
    try:
        data = request.get_json()
        doc_type = data.get('doc_type', 'lookbook')
        page_num = data.get('page_num')
        prompt = data.get('prompt', '').strip()
        use_layout_only = data.get('use_layout_only', False)
        
        logger.info(f"📥 [GENERATE-PAGE] İstek alındı - Doc Type: {doc_type}, Sayfa: {page_num}, Layout Only: {use_layout_only}, Prompt uzunluğu: {len(prompt)} karakter")
        
        original_img = None
        
        # Layout-only modunda sayfa numarası gerekmez
        if not use_layout_only:
            if not page_num:
                logger.warning(f"⚠️ [GENERATE-PAGE] Sayfa numarası eksik")
                return jsonify({'success': False, 'error': 'Sayfa numarası gerekli'}), 400
            
            pdf_path = LOOKBOOK_PDF if doc_type == 'lookbook' else LINESHEET_PDF
            
            logger.info(f"📄 [GENERATE-PAGE] PDF yolu: {pdf_path}")
            
            if not os.path.exists(pdf_path):
                logger.error(f"❌ [GENERATE-PAGE] PDF dosyası bulunamadı: {pdf_path}")
                return jsonify({'success': False, 'error': 'PDF dosyası bulunamadı'}), 404
            
            # Sadece ilgili sayfayı PDF'den çıkar - yüksek çözünürlükte
            logger.info(f"🔄 [GENERATE-PAGE] PDF'den sadece sayfa {page_num} çıkarılıyor (dpi=600 - yüksek çözünürlük)...")
            page_index = page_num - 1
            
            # Sadece ilgili sayfayı çıkar - 600 DPI ile yüksek çözünürlükte
            # 600 DPI = yaklaşık 4960x7016 piksel (A4 için)
            images = convert_from_path(pdf_path, dpi=600, first_page=page_num, last_page=page_num)
            
            if not images or len(images) == 0:
                logger.error(f"❌ [GENERATE-PAGE] Sayfa {page_num} çıkarılamadı")
                return jsonify({'success': False, 'error': 'Geçersiz sayfa numarası'}), 400
            
            original_img = images[0]  # Sadece bir sayfa çıkarıldığı için ilk eleman
            logger.info(f"✅ [GENERATE-PAGE] Sayfa {page_num} başarıyla çıkarıldı")
            logger.info(f"📸 [GENERATE-PAGE] Orijinal görsel boyutu: {original_img.size}")
        
        # Prompt yoksa ve layout-only değilse orijinal görseli döndür
        if not prompt and not use_layout_only and original_img:
            logger.info(f"ℹ️ [GENERATE-PAGE] Prompt boş, orijinal görsel döndürülüyor")
            img_buffer = BytesIO()
            original_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            logger.info(f"✅ [GENERATE-PAGE] Orijinal görsel başarıyla döndürüldü")
            return jsonify({
                'success': True,
                'image': f"data:image/png;base64,{img_base64}"
            })
        
        # Gemini API hazırlığı
        if not GEMINI_API_KEY:
            logger.error(f"❌ [GENERATE-PAGE] Google AI API key yapılandırılmamış")
            raise Exception("Google AI API key yapılandırılmamış")
        
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        logger.info(f"🖼️ [GENERATE-PAGE] Sayfa {page_num} Gemini API ile düzenleniyor...")
        logger.info(f"📝 [GENERATE-PAGE] Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"📝 [GENERATE-PAGE] Prompt: {prompt}")
        
        aspect_ratio = data.get('aspect_ratio') or '16:9'
        if aspect_ratio not in ['16:9', '9:16']:
            logger.warning(f"⚠️ [GENERATE-PAGE] Geçersiz aspect_ratio '{aspect_ratio}' alındı, 16:9 kullanılacak")
            aspect_ratio = '16:9'
        
        # Gelişmiş prompt oluştur - High-fidelity preservation için
        full_prompt = f"""You are a professional fashion photography AI. Follow these CRITICAL RULES EXACTLY:

═══════════════════════════════════════════════════════════════
1. PRESERVE MODEL FACE - ABSOLUTE PRIORITY:
═══════════════════════════════════════════════════════════════
   - The model's FACE from the reference image MUST remain 100% IDENTICAL
   - Facial features, skin tone, expression, hair style, hair color - PRESERVE ALL
   - Face shape, eye color, nose, lips, facial structure - DO NOT CHANGE
   - The model in output MUST be the SAME person as in the input reference
   - This is NON-NEGOTIABLE - face preservation is the TOP priority

═══════════════════════════════════════════════════════════════
2. PRESERVE LAYOUT STRUCTURE - CRITICAL:
═══════════════════════════════════════════════════════════════
   - If a layout image is provided, PRESERVE its exact structure:
     • Grid layout and divisions MUST remain unchanged
     • Text elements, logos, typography - keep EXACTLY as shown
     • Borders, frames, spacing - maintain precisely
     • Multi-grid layouts: respect each grid's composition
   - The model MUST stay WITHIN the designated grid boundaries
   - DO NOT let the model overflow or break the grid structure

═══════════════════════════════════════════════════════════════
3. PRESERVE GARMENT/CLOTHING EXACTLY:
═══════════════════════════════════════════════════════════════
   - Use the EXACT clothing design from the layout image
   - Colors, patterns, textures, cuts, details - match PERFECTLY
   - Fabric draping and fit should look natural on the model's body
   - Maintain all garment details: buttons, zippers, seams, embellishments

═══════════════════════════════════════════════════════════════
4. APPLY USER STYLING INSTRUCTIONS:
═══════════════════════════════════════════════════════════════
{prompt}

IMPORTANT STYLING GUIDELINES:
   - Location/Setting: Create the specified environment (street, studio, etc.)
   - Lighting: Match the described lighting conditions (sunny, studio, golden hour)
   - Camera Framing: Follow specified framing (close-up, full body, waist-up)
     • Close-up/Tight crop: Head to chest/waist visible
     • Medium shot: Head to waist/hips visible
     • Full body: Entire person visible
   - Accessories: Add specified items (jewelry, bags, shoes) naturally
   - Background: Create the described background, remove people if specified
   - Pose: Natural, fashion-appropriate poses for each grid
   - Season/Weather: Reflect the specified season and weather conditions

═══════════════════════════════════════════════════════════════
5. COMPOSITION & QUALITY:
═══════════════════════════════════════════════════════════════
   - Professional high-fashion photography quality
   - Natural, realistic lighting and shadows
   - Sharp focus on the model, appropriate depth of field
   - Color grading appropriate for fashion editorial
   - Maintain spatial awareness and realistic proportions

═══════════════════════════════════════════════════════════════
EXECUTION CHECKLIST:
═══════════════════════════════════════════════════════════════
✓ Model's face is IDENTICAL to reference
✓ Layout structure is PRESERVED (grids, text, borders)
✓ Garment matches layout EXACTLY
✓ Model stays WITHIN grid boundaries
✓ All user styling instructions are applied
✓ Professional fashion photography quality
✓ Natural and realistic result

Generate the high-quality fashion image now, following ALL rules above."""
        
        # Kullanıcıdan gelen model yüzlerini oku (tek veya çoklu destek)
        model_faces_payload = data.get('model_faces') or []
        single_face = data.get('model_face')
        if single_face:
            model_faces_payload.append(single_face)

        model_face_images = []
        if model_faces_payload:
            logger.info(f"🧑‍🎨 [GENERATE-PAGE] {len(model_faces_payload)} adet model yüzü alındı, Gemini'ya eklenecek")
            for idx, face in enumerate(model_faces_payload):
                try:
                    face_data = face.get('data')
                    if not face_data:
                        logger.warning(f"⚠️ [GENERATE-PAGE] Model yüzü #{idx+1} boş veri içeriyor, atlanıyor")
                        continue
                    if ',' in face_data:
                        face_data = face_data.split(',', 1)[1]
                    face_bytes = base64.b64decode(face_data)
                    face_img = Image.open(BytesIO(face_bytes))
                    model_face_images.append(face_img)
                except Exception as face_err:
                    logger.warning(f"⚠️ [GENERATE-PAGE] Model yüzü #{idx+1} işlenemedi: {str(face_err)}")

        # Layout görselini işle
        layout_payload = data.get('layout')
        layout_image = None
        if layout_payload:
            try:
                layout_data = layout_payload.get('data')
                layout_type = layout_payload.get('type', 'png')
                layout_name = layout_payload.get('name', 'layout')
                
                logger.info(f"📐 [GENERATE-PAGE] Layout görseli alındı - Tip: {layout_type}, İsim: {layout_name}")
                
                if layout_type == 'pdf':
                    # PDF ise, ilk sayfayı görsel olarak çıkar
                    if ',' in layout_data:
                        layout_data = layout_data.split(',', 1)[1]
                    pdf_bytes = base64.b64decode(layout_data)
                    
                    # PDF'i geçici dosyaya kaydet
                    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    temp_pdf.write(pdf_bytes)
                    temp_pdf.close()
                    
                    try:
                        # PDF'in ilk sayfasını görsel olarak çıkar
                        pdf_images = convert_from_path(temp_pdf.name, dpi=600, first_page=1, last_page=1)
                        if pdf_images and len(pdf_images) > 0:
                            layout_image = pdf_images[0]
                            logger.info(f"✅ [GENERATE-PAGE] Layout PDF'den görsel çıkarıldı ({layout_image.size})")
                        else:
                            logger.warning(f"⚠️ [GENERATE-PAGE] Layout PDF'den görsel çıkarılamadı")
                    finally:
                        # Geçici dosyayı sil
                        try:
                            os.unlink(temp_pdf.name)
                        except:
                            pass
                else:
                    # PNG, JPEG veya diğer görsel formatları direkt kullan
                    if ',' in layout_data:
                        layout_data = layout_data.split(',', 1)[1]
                    layout_bytes = base64.b64decode(layout_data)
                    layout_image = Image.open(BytesIO(layout_bytes))
                    logger.info(f"✅ [GENERATE-PAGE] Layout görseli yüklendi - Format: {layout_type}, Boyut: {layout_image.size}")
            except Exception as layout_err:
                logger.warning(f"⚠️ [GENERATE-PAGE] Layout görseli işlenemedi: {str(layout_err)}")

        # Fashion use case için optimal model seçimi
        # Eğer hem manken yüzü hem layout varsa, Gemini 3 Pro kullan (daha iyi high-fidelity preservation)
        use_gemini_3_pro = (len(model_face_images) > 0 and layout_image is not None)
        selected_model = "gemini-3-pro-image-preview" if use_gemini_3_pro else GEMINI_MODEL_ID
        
        logger.info(f"🚀 [GENERATE-PAGE] Gemini API'ye istek gönderiliyor ({selected_model})...")
        logger.info(f"🖼️ [GENERATE-PAGE] Image size: 4K (4096x4096), Aspect Ratio: {aspect_ratio}")
        if model_face_images:
            logger.info(f"🧩 [GENERATE-PAGE] İçerik listesine {len(model_face_images)} model yüzü eklendi")
        if layout_image:
            logger.info(f"📐 [GENERATE-PAGE] İçerik listesine layout görseli eklendi")
        
        # Görsel sıralaması önemli: prompt -> manken yüzü -> layout -> orijinal sayfa (varsa)
        contents = [full_prompt]
        contents.extend(model_face_images)
        if layout_image:
            contents.append(layout_image)
        
        # Layout-only modunda orijinal görsel ekleme
        if not use_layout_only and original_img:
            contents.append(original_img)
        elif use_layout_only:
            logger.info(f"📐 [GENERATE-PAGE] Layout-only modu: Sadece layout görseli kullanılıyor")

        # 4K çözünürlüklü görsel için ImageConfig kullan
        response = client.models.generate_content(
            model=selected_model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE'],
                image_config=types.ImageConfig(
                    image_size="4K",  # 4096x4096
                    aspect_ratio=aspect_ratio
                )
            )
        )
        logger.info(f"📥 [GENERATE-PAGE] Gemini API'den yanıt alındı")
        
        # Düzenlenmiş görseli al (yeni API yapısı)
        edited_img = None
        
        # Yeni API response yapısını kontrol et
        if hasattr(response, 'parts'):
            # Yeni API: response.parts kullan
            for part in response.parts:
                if hasattr(part, 'as_image'):
                    try:
                        image = part.as_image()
                        if image:
                            # Image objesini PIL Image'e çevir
                            img_bytes = image.read()
                            edited_img = Image.open(BytesIO(img_bytes))
                            logger.info(f"✅ [GENERATE-PAGE] Görsel yeni API formatından alındı")
                            break
                    except Exception as e:
                        logger.warning(f"⚠️ [GENERATE-PAGE] Görsel parse edilemedi: {str(e)}")
                elif hasattr(part, 'inline_data') and part.inline_data:
                    # Eski format: inline_data
                    try:
                        edited_img = Image.open(BytesIO(part.inline_data.data))
                        logger.info(f"✅ [GENERATE-PAGE] Görsel inline_data'dan alındı")
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ [GENERATE-PAGE] inline_data parse edilemedi: {str(e)}")
        
        # Eski API formatını da kontrol et (fallback)
        if not edited_img and hasattr(response, 'candidates') and response.candidates:
            if response.candidates[0].content.parts:
                image_parts = [
                    part.inline_data.data
                    for part in response.candidates[0].content.parts
                    if hasattr(part, 'inline_data') and part.inline_data
                ]
                if image_parts:
                    try:
                        edited_img = Image.open(BytesIO(image_parts[0]))
                        logger.info(f"✅ [GENERATE-PAGE] Görsel eski API formatından alındı")
                    except Exception as e:
                        logger.warning(f"⚠️ [GENERATE-PAGE] Görsel decode edilemedi: {str(e)}")
        
        # Görsel alınamadıysa orijinali kullan
        if not edited_img:
            logger.warning(f"⚠️ [GENERATE-PAGE] Sayfa {page_num} için görsel alınamadı, orijinal kullanılıyor")
            edited_img = original_img
        
        # Görseli base64'e çevir
        logger.info(f"🔄 [GENERATE-PAGE] Düzenlenmiş görsel base64'e çevriliyor...")
        img_buffer = BytesIO()
        edited_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        logger.info(f"✅ [GENERATE-PAGE] Sayfa {page_num} başarıyla düzenlendi ve base64'e çevrildi (Boyut: {len(img_base64)} karakter)")
        
        return jsonify({
            'success': True,
            'image': f"data:image/png;base64,{img_base64}"
        })
        
    except Exception as e:
        logger.error(f"❌ [GENERATE-PAGE] Sayfa düzenleme hatası: {str(e)}")
        logger.error(f"❌ [GENERATE-PAGE] Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/collection-builder/generate', methods=['POST'])
def generate_collection():
    """Düzenlenmiş görselleri PDF'e çevir"""
    temp_pdf_path = None
    try:
        data = request.get_json()
        if not data:
            logger.error("❌ [GENERATE] JSON verisi alınamadı")
            return jsonify({'success': False, 'error': 'JSON verisi gerekli'}), 400
        
        edited_images_data = data.get('edited_images', [])
        
        if not edited_images_data:
            logger.error("❌ [GENERATE] Düzenlenmiş görsel bulunamadı")
            return jsonify({'success': False, 'error': 'Düzenlenmiş görsel bulunamadı'}), 400
        
        logger.info(f"📄 [GENERATE] PDF oluşturuluyor: {len(edited_images_data)} sayfa")
        
        # Base64 görselleri PIL Image'e çevir
        edited_images = []
        for i, img_data in enumerate(edited_images_data):
            try:
                if not img_data:
                    logger.warning(f"⚠️ [GENERATE] Sayfa {i+1} için boş görsel verisi")
                    continue
                    
                # Base64 string'i temizle
                if isinstance(img_data, str):
                    if img_data.startswith('data:image'):
                        img_data = img_data.split(',')[1]
                    
                    # Base64 decode
                    try:
                        img_bytes = base64.b64decode(img_data)
                    except Exception as decode_error:
                        logger.error(f"❌ [GENERATE] Sayfa {i+1} base64 decode hatası: {str(decode_error)}")
                        raise Exception(f"Sayfa {i+1} görseli geçersiz format")
                    
                    # PIL Image'e çevir
                    try:
                        img = Image.open(BytesIO(img_bytes))
                        edited_images.append(img)
                        logger.info(f"✅ [GENERATE] Sayfa {i+1} başarıyla yüklendi ({img.size[0]}x{img.size[1]})")
                    except Exception as img_error:
                        logger.error(f"❌ [GENERATE] Sayfa {i+1} görsel açma hatası: {str(img_error)}")
                        raise Exception(f"Sayfa {i+1} görseli açılamadı")
                else:
                    logger.warning(f"⚠️ [GENERATE] Sayfa {i+1} için geçersiz veri tipi: {type(img_data)}")
            except Exception as page_error:
                logger.error(f"❌ [GENERATE] Sayfa {i+1} işleme hatası: {str(page_error)}")
                # Devam et, diğer sayfaları işle
                continue
        
        if not edited_images:
            logger.error("❌ [GENERATE] Hiç geçerli görsel bulunamadı")
            return jsonify({'success': False, 'error': 'Hiç geçerli görsel bulunamadı'}), 400
        
        logger.info(f"✅ [GENERATE] {len(edited_images)} görsel başarıyla yüklendi, PDF oluşturuluyor...")
        
        # PDF oluştur
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_pdf_path = temp_pdf.name
        temp_pdf.close()
        
        # ReportLab ile PDF oluştur
        try:
            c = canvas.Canvas(temp_pdf_path, pagesize=A4)
            
            for i, img in enumerate(edited_images):
                try:
                    # Görseli PDF boyutuna uyarla
                    img_width, img_height = img.size
                    page_width, page_height = A4
                    
                    # Aspect ratio koru
                    scale = min(page_width / img_width, page_height / img_height)
                    new_width = img_width * scale
                    new_height = img_height * scale
                    
                    # Ortala
                    x = (page_width - new_width) / 2
                    y = (page_height - new_height) / 2
                    
                    img_buffer = BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    c.drawImage(ImageReader(img_buffer), x, y, width=new_width, height=new_height)
                    c.showPage()
                    logger.info(f"✅ [GENERATE] Sayfa {i+1} PDF'e eklendi")
                except Exception as page_error:
                    logger.error(f"❌ [GENERATE] Sayfa {i+1} PDF'e eklenirken hata: {str(page_error)}")
                    # Devam et, diğer sayfaları ekle
                    continue
            
            c.save()
            logger.info(f"✅ [GENERATE] PDF başarıyla oluşturuldu: {temp_pdf_path}")
            
        except Exception as pdf_error:
            logger.error(f"❌ [GENERATE] PDF oluşturma hatası: {str(pdf_error)}")
            raise Exception(f"PDF oluşturulamadı: {str(pdf_error)}")
        
        # PDF'i base64 olarak döndür
        try:
            with open(temp_pdf_path, 'rb') as f:
                pdf_data = f.read()
                pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
            logger.info(f"✅ [GENERATE] PDF base64'e çevrildi (uzunluk: {len(pdf_base64)} karakter)")
        except Exception as read_error:
            logger.error(f"❌ [GENERATE] PDF okuma hatası: {str(read_error)}")
            raise Exception(f"PDF okunamadı: {str(read_error)}")
        
        # Temp dosyayı sil
        try:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)
                logger.info(f"✅ [GENERATE] Temp dosya silindi: {temp_pdf_path}")
        except Exception as cleanup_error:
            logger.warning(f"⚠️ [GENERATE] Temp dosya silinemedi: {str(cleanup_error)}")
        
        return jsonify({
            'success': True,
            'pdf_data': f"data:application/pdf;base64,{pdf_base64}"
        })
        
    except Exception as e:
        logger.error(f"❌ [GENERATE] PDF oluşturma hatası: {str(e)}")
        logger.error(f"❌ [GENERATE] Traceback: {traceback.format_exc()}")
        
        # Temp dosyayı temizle
        try:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)
        except:
            pass
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

