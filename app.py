# Enable eventlet monkey patching before any other imports
import eventlet
eventlet.monkey_patch()

import os
import time
import cv2
import numpy as np
import base64
import speech_recognition as sr
import pytesseract
import logging

from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_socketio import SocketIO, emit
from deep_translator import GoogleTranslator
from gtts import gTTS
from pydub import AudioSegment
from docx import Document
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
from ai_ocr import enhance_text_with_ai, get_ocr_tool_references
from offline_ocr import enhance_text_offline

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")
# Initialize SocketIO with eventlet async mode
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", logger=True, engineio_logger=True)

logger.info("Initializing application with eventlet WebSocket support")

# Existing configuration 
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'doc', 'docx', 'wav', 'mp3'}

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'static'), exist_ok=True)
os.makedirs('static', exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise

def extract_text_from_file(file_path):
    """Extract text based on file type"""
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    file_ext = file_path.rsplit('.', 1)[1].lower()
    logger.debug(f"Processing file with extension: {file_ext}")

    try:
        if file_ext in ['pdf']:
            text = extract_text_from_pdf(file_path)
            return {'text': text, 'confidence': 1.0, 'method': 'pdf'}
        elif file_ext in ['doc', 'docx']:
            text = extract_text_from_docx(file_path)
            return {'text': text, 'confidence': 1.0, 'method': 'docx'}
        elif file_ext in ['png', 'jpg', 'jpeg', 'gif']:
            # For images, try multiple preprocessing methods
            best_result = None
            highest_confidence = 0

            methods = ['default', 'denoised', 'sharpened', 'high_contrast']
            for method in methods:
                try:
                    logger.debug(f"Trying OCR with method: {method}")
                    preprocessed_path = preprocess_image(file_path, method)

                    if not os.path.exists(preprocessed_path):
                        logger.warning(f"Preprocessed image not found: {preprocessed_path}")
                        continue

                    # Perform OCR and get data with confidence scores
                    text = pytesseract.image_to_string(
                        preprocessed_path,
                        config='--oem 3 --psm 6'
                    )

                    # Get confidence scores
                    data = pytesseract.image_to_data(
                        preprocessed_path,
                        config='--oem 3 --psm 6',
                        output_type=pytesseract.Output.DICT
                    )

                    conf_scores = [float(conf) for conf in data['conf'] if conf != '-1']
                    if conf_scores:
                        avg_confidence = sum(conf_scores) / len(conf_scores)

                        # Clean up text
                        text = ' '.join(text.split())  # Remove extra whitespace

                        logger.debug(f"Method {method} confidence: {avg_confidence}, text length: {len(text)}")

                        if avg_confidence > highest_confidence and text.strip():
                            highest_confidence = avg_confidence
                            best_result = {
                                'text': text,
                                'confidence': avg_confidence / 100,  # Convert to 0-1 range
                                'method': method
                            }

                    # Clean up preprocessed image
                    if os.path.exists(preprocessed_path):
                        os.remove(preprocessed_path)

                except Exception as e:
                    logger.error(f"Error processing with method {method}: {str(e)}")
                    continue

            if best_result is None or not best_result['text'].strip():
                raise ValueError("Failed to extract any text from the image")

            return best_result
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    except Exception as e:
        logger.error(f"Error extracting text from file: {str(e)}")
        raise

def preprocess_image(image_path, method='default'):
    """Preprocess image for better OCR accuracy"""
    try:
        # Validate input
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")

        # Read image with multiple attempts
        image = None
        try:
            image = cv2.imread(image_path)
        except Exception as e:
            logger.error(f"Failed to read image with cv2.imread: {str(e)}")

        if image is None:
            try:
                # Try reading with numpy first
                image = np.fromfile(image_path, np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"Failed to read image with numpy: {str(e)}")
                raise ValueError(f"Could not read image file: {image_path}")

        # Validate image data
        if image is None or image.size == 0:
            raise ValueError("Invalid image data")

        # Log original image properties
        logger.debug(f"Original image shape: {image.shape}")
        logger.debug(f"Original image dtype: {image.dtype}")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize if too large while maintaining aspect ratio
        max_dimension = 3200  # Increased for better quality
        height, width = gray.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Apply different preprocessing methods
        if method == 'default':
            # Basic preprocessing
            processed = cv2.medianBlur(gray, 3)  # Remove noise
            processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        elif method == 'denoised':
            # Heavy denoising
            processed = cv2.fastNlMeansDenoising(gray)
            processed = cv2.GaussianBlur(processed, (3,3), 0)
            processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        elif method == 'sharpened':
            # Sharpen edges
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            processed = cv2.filter2D(gray, -1, kernel)
            processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        elif method == 'high_contrast':
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            processed = clahe.apply(gray)
            processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Save preprocessed image
        output_dir = os.path.dirname(image_path)
        preprocessed_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_preprocessed_{method}.png"
        preprocessed_path = os.path.join(output_dir, preprocessed_filename)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save image with error checking
        success = cv2.imwrite(preprocessed_path, processed)
        if not success:
            raise ValueError(f"Failed to save preprocessed image to {preprocessed_path}")

        logger.debug(f"Successfully saved preprocessed image: {preprocessed_path}")
        return preprocessed_path

    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        raise

def perform_ocr(image_path):
    try:
        # Optimized tesseract configuration for multiple languages and improved accuracy
        custom_config = (
            '--oem 3 '  # Use LSTM OCR Engine Mode
            '--psm 6 '  # Assume uniform block of text
            '-l eng+fra+spa+deu+ita+por+rus+chi_sim+jpn+kor+ara+hin+ben+tur+vie+tha+nld+pol+ell+heb'  # Multiple language support
            '-c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()" '
            '-c textord_heavy_nr=1 '  # More aggressive noise removal
            '-c textord_min_linesize=2.5 '  # Better line detection
            '-c tessedit_do_invert=0 '  # Don't invert colors
            '-c tessedit_image_border=20 '  # Add border around text
            '-c tessedit_unrej_any_wd=1 '  # Less strict word rejection
            '-c tessedit_write_images=0'  # Don't write debug images
        )

        # Perform OCR with multiple preprocessing methods
        results = []
        methods = ['default', 'denoised', 'sharpened', 'high_contrast']

        for method in methods:
            preprocessed_path = preprocess_image(image_path, method)
            text = pytesseract.image_to_string(
                preprocessed_path,
                config=custom_config
            )

            # Get confidence scores
            data = pytesseract.image_to_data(preprocessed_path, config=custom_config, output_type=pytesseract.Output.DICT)
            confidence_scores = [float(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

            results.append({
                'text': text,
                'confidence': avg_confidence,
                'method': method
            })

            # Clean up preprocessed image
            if os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)

        # Select best result based on confidence score
        best_result = max(results, key=lambda x: x['confidence'])

        # Post-process the text
        processed_text = ' '.join(best_result['text'].split())
        processed_text = processed_text.replace('|', 'I')  # Common mistake: pipe for capital I
        processed_text = processed_text.replace('0', 'O')  # Common mistake: zero for capital O
        processed_text = processed_text.replace('1', 'l')  # Common mistake: one for lowercase L
        processed_text = processed_text.replace('\n', ' ')

        return {
            'text': processed_text.strip(),
            'confidence': best_result['confidence'],
            'method': best_result['method']
        }
    except Exception as e:
        logger.error(f"Error performing OCR: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not supported"}), 400

    try:
        # Create necessary directories
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs('static', exist_ok=True)

        # Save file with secure filename
        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save with error handling
        try:
            file.save(file_path)
            if not os.path.exists(file_path):
                raise ValueError("File failed to save")
            logger.debug(f"Successfully saved file: {file_path}")
        except Exception as e:
            raise ValueError(f"Error saving file: {str(e)}")

        # Initialize preview path
        preview_path = None
        response_data = {}

        # Handle different file types
        file_ext = filename.rsplit('.', 1)[1].lower()

        if file_ext in ['png', 'jpg', 'jpeg', 'gif']:
            # For images, move to static for preview
            preview_path = os.path.join('static', filename)
            os.rename(file_path, preview_path)
            file_path = preview_path
            response_data["preview_url"] = url_for('static', filename=filename)
        elif file_ext in ['pdf', 'doc', 'docx']:
            # For documents, just keep in uploads folder
            response_data["file_name"] = filename

        # Extract text from the file with enhanced OCR
        try:
            ocr_result = extract_text_from_file(file_path)
            logger.debug(f"OCR Result: {ocr_result}")  # Add debug logging

            response_data.update({
                "text": ocr_result['text'],
                "confidence": ocr_result.get('confidence', 0),
                "method": ocr_result.get('method', 'default')
            })
        except Exception as e:
            logger.error(f"OCR processing error: {str(e)}")
            raise ValueError(f"Error processing text: {str(e)}")

        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Only remove the file if it's not an image preview
        if file_path and preview_path is None and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error cleaning up file: {str(e)}")

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    filename = f"{int(time.time())}_{audio_file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        audio_file.save(file_path)

        # Convert MP3 to WAV if necessary
        if file_path.endswith('.mp3'):
            sound = AudioSegment.from_mp3(file_path)
            wav_path = file_path.replace('.mp3', '.wav')
            sound.export(wav_path, format="wav")
            file_path = wav_path

        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)

        # Store audio for preview
        preview_path = os.path.join('static', filename)
        os.rename(file_path, preview_path)

        response_data = {
            "text": text,
            "audio_url": url_for('static', filename=filename)
        }

        return jsonify(response_data)
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 500
    except sr.RequestError as e:
        return jsonify({"error": f"Error with speech recognition service: {e}"}), 500
    except Exception as e:
        logger.error(f"Error processing speech: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        if file_path.endswith('.mp3') and os.path.exists(file_path.replace('.mp3', '.wav')):
            os.remove(file_path.replace('.mp3', '.wav'))


@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.json
    text = data.get("text", "")
    target_lang = data.get("target_lang", "en")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        # Get list of supported languages
        supported_languages = {
            'af': 'Afrikaans', 'sq': 'Albanian', 'am': 'Amharic', 'ar': 'Arabic',
            'hy': 'Armenian', 'az': 'Azerbaijani', 'eu': 'Basque', 'be': 'Belarusian',
            'bn': 'Bengali', 'bs': 'Bosnian', 'bg': 'Bulgarian', 'ca': 'Catalan',
            'ceb': 'Cebuano', 'zh-CN': 'Chinese (Simplified)', 'zh-TW': 'Chinese (Traditional)',
            'co': 'Corsican', 'hr': 'Croatian', 'cs': 'Czech', 'da': 'Danish',
            'nl': 'Dutch', 'en': 'English', 'eo': 'Esperanto', 'et': 'Estonian',
            'fi': 'Finnish', 'fr': 'French', 'fy': 'Frisian', 'gl': 'Galician',
            'ka': 'Georgian', 'de': 'German', 'el': 'Greek', 'gu': 'Gujarati',
            'ht': 'Haitian Creole', 'ha': 'Hausa', 'haw': 'Hawaiian', 'he': 'Hebrew',
            'hi': 'Hindi', 'hmn': 'Hmong', 'hu': 'Hungarian', 'is': 'Icelandic',
            'ig': 'Igbo', 'id': 'Indonesian', 'ga': 'Irish', 'it': 'Italian',
            'ja': 'Japanese', 'jv': 'Javanese', 'kn': 'Kannada', 'kk': 'Kazakh',
            'km': 'Khmer', 'rw': 'Kinyarwanda', 'ko': 'Korean', 'ku': 'Kurdish',
            'ky': 'Kyrgyz', 'lo': 'Lao', 'la': 'Latin', 'lv': 'Latvian',
            'lt': 'Lithuanian', 'lb': 'Luxembourgish', 'mk': 'Macedonian',
            'mg': 'Malagasy', 'ms': 'Malay', 'ml': 'Malayalam', 'mt': 'Maltese',
            'mi': 'Maori', 'mr': 'Marathi', 'mn': 'Mongolian', 'my': 'Myanmar (Burmese)',
            'ne': 'Nepali', 'no': 'Norwegian', 'ny': 'Nyanja (Chichewa)',
            'or': 'Odia (Oriya)', 'ps': 'Pashto', 'fa': 'Persian', 'pl': 'Polish',
            'pt': 'Portuguese', 'pa': 'Punjabi', 'ro': 'Romanian', 'ru': 'Russian',
            'sm': 'Samoan', 'gd': 'Scots Gaelic', 'sr': 'Serbian', 'st': 'Sesotho',
            'sn': 'Shona', 'sd': 'Sindhi', 'si': 'Sinhala (Sinhalese)', 'sk': 'Slovak',
            'sl': 'Slovenian', 'so': 'Somali', 'es': 'Spanish', 'su': 'Sundanese',
            'sw': 'Swahili', 'sv': 'Swedish', 'tl': 'Tagalog (Filipino)', 'tg': 'Tajik',
            'ta': 'Tamil', 'tt': 'Tatar', 'te': 'Telugu', 'th': 'Thai', 'tr': 'Turkish',
            'tk': 'Turkmen', 'uk': 'Ukrainian', 'ur': 'Urdu', 'ug': 'Uyghur',
            'uz': 'Uzbek', 'vi': 'Vietnamese', 'cy': 'Welsh', 'xh': 'Xhosa',
            'yi': 'Yiddish', 'yo': 'Yoruba', 'zu': 'Zulu'
        }

        # Validate target language
        if target_lang not in supported_languages:
            return jsonify({
                "error": f"Unsupported target language: {target_lang}. Please choose from the supported languages: {', '.join(supported_languages.keys())}"
            }), 400

        # Use GoogleTranslator with improved error handling
        translator = GoogleTranslator(source='auto', target=target_lang)

        # Split long text into chunks if needed (API limit)
        max_chunk_size = 5000
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

        translated_chunks = []
        for chunk in chunks:
            translated_chunk = translator.translate(chunk)
            if not translated_chunk:
                raise ValueError(f"Translation failed for chunk: {chunk[:100]}...")
            translated_chunks.append(translated_chunk)

        translated_text = ' '.join(translated_chunks)

        return jsonify({
            "translated_text": translated_text,
            "source_language": translator._source,
            "target_language": target_lang,
            "language_name": supported_languages[target_lang]
        })
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    text = request.json.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        # Create static directory if it doesn't exist
        os.makedirs('static', exist_ok=True)

        # Generate unique filename
        filename = f"{int(time.time())}.mp3"
        file_path = os.path.join('static', filename)

        # Generate speech
        tts = gTTS(text)
        tts.save(file_path)

        # Return URL for the audio file
        return jsonify({"audio_url": url_for('static', filename=filename)})
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Update the live_ocr route to include AI enhancement
@app.route('/live_ocr', methods=['POST'])
def live_ocr():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = data.get("image")
        method = data.get("method", "default")

        if not image_data:
            return jsonify({"error": "Empty image data"}), 400

        try:
            # Decode base64 image
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image data")

            # Save temporary file
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"live_{int(time.time())}.jpg")
            cv2.imwrite(temp_path, img)

            # Process and perform OCR
            preprocessed_path = preprocess_image(temp_path, method)
            extracted_text = perform_ocr(preprocessed_path)

            # Log success for debugging
            logger.debug(f"Successfully processed image with method: {method}")
            logger.debug(f"Extracted text length: {len(extracted_text)}")

            try:
                # Try AI enhancement first
                enhanced_result = enhance_text_with_ai(extracted_text)
                response = {
                    "text": enhanced_result["enhanced_text"],
                    "original_text": extracted_text,
                    "confidence_score": enhanced_result["confidence_score"],
                    "ai_powered": True
                }
            except Exception as e:
                # Fallback to offline enhancement if AI service is unavailable
                logger.warning(f"Falling back to offline processing due to AI service error: {str(e)}")
                enhanced_result = enhance_text_offline(extracted_text)
                response = {
                    "text": enhanced_result["enhanced_text"],
                    "original_text": extracted_text,
                    "confidence_score": enhanced_result["confidence_score"],
                    "ai_powered": False,
                    "offline_mode": True
                }

            return jsonify(response)

        except ValueError as e:
            logger.error(f"Image processing error: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error in live OCR: {str(e)}")
            if "insufficient_quota" in str(e):
                # Fallback to offline enhancement
                enhanced_result = enhance_text_offline(extracted_text)
                return jsonify({
                    "text": enhanced_result["enhanced_text"],
                    "original_text": extracted_text,
                    "confidence_score": enhanced_result["confidence_score"],
                    "ai_powered": False,
                    "offline_mode": True
                })
            return jsonify({"error": "Failed to process image"}), 500
        finally:
            # Clean up temporary files
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            if 'preprocessed_path' in locals() and os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)

    except Exception as e:
        logger.error(f"Unexpected error in live OCR: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/rescan', methods=['POST'])
def rescan_image():
    try:
        data = request.json
        image_path = os.path.join('static', data.get('filename'))
        method = data.get('method', 'default')

        if not os.path.exists(image_path):
            return jsonify({"error": "Image not found"}), 404

        preprocessed_path = preprocess_image(image_path, method)
        extracted_text = perform_ocr(preprocessed_path)

        # Cleanup preprocessed image
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)

        return jsonify({"text": extracted_text, "method": method})
    except Exception as e:
        logger.error(f"Error in rescan: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/ocr_tools', methods=['GET'])
def get_ocr_tools():
    """Endpoint to get information about OCR tools"""
    return jsonify(get_ocr_tool_references())

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info("Client connected")
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info("Client disconnected")

@socketio.on('start_transcription')
def handle_transcription_start():
    """Handle start of live transcription"""
    logger.info("Live transcription started")
    emit('transcription_status', {'status': 'started'})

@socketio.on('audio_data')
def handle_audio_data(data):
    """Process incoming audio data chunks"""
    try:
        # Get audio data from the event
        audio_data = data.get('audio')

        if not audio_data:
            return

        # Save temporary audio file
        temp_filename = f"temp_audio_{time.time()}.wav"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)

        try:
            # Decode and save audio data
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)

            # Initialize speech recognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source)
                # Record audio from file
                audio = recognizer.record(source)

            try:
                # Attempt to recognize speech
                text = recognizer.recognize_google(audio)
                logger.debug(f"Transcribed text: {text}")
                emit('transcription_result', {
                    'text': text,
                    'final': True
                })
            except sr.UnknownValueError:
                logger.warning("Could not understand audio")
                emit('transcription_result', {
                    'text': '',
                    'error': 'Could not understand audio'
                })
            except sr.RequestError as e:
                logger.error(f"Speech recognition service error: {str(e)}")
                emit('transcription_result', {
                    'text': '',
                    'error': f'Error with speech recognition service: {str(e)}'
                })

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logger.error(f"Error processing audio data: {str(e)}")
        emit('transcription_result', {
            'text': '',
            'error': 'Error processing audio'
        })

@socketio.on('stop_transcription')
def handle_transcription_stop():
    """Handle end of live transcription"""
    logger.info("Live transcription stopped")
    emit('transcription_status', {'status': 'stopped'})

if __name__ == '__main__':
    logger.info("Starting server with eventlet WebSocket support")
    # ALWAYS serve the app on port 5000
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=True, log_output=True)