import os
import base64
import logging  # Add this missing import
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_session import Session
from dotenv import load_dotenv
from nlp.enhanced_diagnostic_processor import EnhancedDiagnosticProcessor
from speech.speech_handler import transcribe_audio_file, get_speech_status
from speech.speech_utils import AudioValidator, AudioProcessor
from youtube.youtube_handler import search_diagnostic_videos, search_manual_videos
from youtube.youtube_config import is_youtube_available, validate_youtube_setup
import tempfile
import math
import pyrebase
from functools import wraps
from firebase_config import FIREBASE_CONFIG
# from PIL import Image
# import io
# import json

# Ensure FFmpeg is accessible (add this function)
def ensure_ffmpeg_path():
    """Ensure FFmpeg is accessible by adding common installation paths"""
    ffmpeg_paths = [
        r'C:\ffmpeg\bin',
        r'C:\Program Files\ffmpeg\bin', 
        r'C:\ProgramData\chocolatey\bin',
        r'C:\tools\ffmpeg\bin'
    ]
    
    for path in ffmpeg_paths:
        if os.path.exists(path):
            if path not in os.environ['PATH']:
                os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
                print(f"✅ Added FFmpeg path to environment: {path}")
            return True
    
    print("❌ FFmpeg not found in common installation paths")
    return False

# Call FFmpeg path setup early
ensure_ffmpeg_path()

# Load environment variables from .env file
load_dotenv()

# Initialize Firebase
try:
    firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
    auth = firebase.auth()
    print("✅ Firebase authentication initialized successfully")
    firebase_available = True
except Exception as e:
    print(f"❌ Firebase initialization failed: {e}")
    print("💡 Please check your firebase_config.py file")
    auth = None
    firebase_available = False

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    llm_available = True
else:
    print("Warning: GOOGLE_API_KEY not found in environment variables")
    llm_available = False

app = Flask(__name__)

# Configure Flask session for authentication
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
app.config['SESSION_PERMANENT'] = False
Session(app)

# Initialize the enhanced diagnostic processor for NLP functionality
try:
    obd_codes_path = os.path.join(os.path.dirname(__file__), 'data', 'obd_codes.json')
    diagnostic_processor = EnhancedDiagnosticProcessor(obd_codes_path)
    print("✅ Enhanced diagnostic processor initialized successfully")
    nlp_available = True
except Exception as e:
    print(f"❌ Error initializing enhanced diagnostic processor: {e}")
    diagnostic_processor = None
    nlp_available = False

# ML Diagnostics Configuration
FEATURES = [
    "Vibration_Amplitude",
    "RMS_Vibration", 
    "Vibration_Frequency",
    "Surface_Temperature",
    "Exhaust_Temperature",
    "Acoustic_dB",
    "Acoustic_Frequency",
    "Intake_Pressure",
    "Exhaust_Pressure",
    "Frequency_Band_Energy",
    "Amplitude_Mean"
]

# Normal operating ranges for rule-based detection
NORMAL_RANGES = {
    "Vibration_Amplitude": (0.1, 0.8),
    "RMS_Vibration": (0.05, 0.25),
    "Vibration_Frequency": (100, 500),
    "Surface_Temperature": (60, 120),
    "Exhaust_Temperature": (200, 600),
    "Acoustic_dB": (50, 85),
    "Acoustic_Frequency": (500, 2000),
    "Intake_Pressure": (0.8, 2.0),
    "Exhaust_Pressure": (0.5, 1.5),
    "Frequency_Band_Energy": (20, 80),
    "Amplitude_Mean": (0.1, 0.6)
}

def predict_anomaly_rule_based(input_data):
    """Rule-based anomaly detection for demo purposes"""
    try:
        anomalies = []
        scores = []
        
        for feature in FEATURES:
            if feature not in input_data:
                return {"error": f"Missing feature: {feature}"}
            
            value = float(input_data[feature])
            min_val, max_val = NORMAL_RANGES[feature]
            
            # Calculate how far outside normal range
            if value < min_val:
                deviation = (min_val - value) / min_val
                anomalies.append(f"{feature} too low ({value:.2f} < {min_val})")
                scores.append(deviation)
            elif value > max_val:
                deviation = (value - max_val) / max_val
                anomalies.append(f"{feature} too high ({value:.2f} > {max_val})")
                scores.append(deviation)
            else:
                # Within normal range
                scores.append(0)
        
        # Calculate overall anomaly score
        max_deviation = max(scores) if scores else 0
        avg_deviation = sum(scores) / len(scores) if scores else 0
        
        # Determine if anomalous
        is_anomaly = len(anomalies) > 0 or max_deviation > 0.2
        anomaly_score = max_deviation + (avg_deviation * 0.3)
        
        # Determine status and risk
        if is_anomaly:
            if max_deviation > 0.5:
                risk_level = "HIGH"
                status = "CRITICAL ANOMALY DETECTED"
            elif max_deviation > 0.3:
                risk_level = "MEDIUM"
                status = "ANOMALY DETECTED"
            else:
                risk_level = "LOW"
                status = "MINOR ANOMALY DETECTED"
        else:
            risk_level = "LOW"
            status = "NORMAL"
        
        return {
            "status": status,
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "confidence": round(max_deviation, 4),
            "risk_level": risk_level,
            "anomalies_detected": anomalies[:3],  # Show top 3 issues
            "total_anomalies": len(anomalies)
        }
        
    except Exception as e:
        return {"error": str(e)}

# Authentication helper functions
def login_required(f):
    """Decorator to require login for protected routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not firebase_available:
            flash('Authentication system is currently unavailable', 'error')
            return redirect(url_for('login'))
        
        if 'user' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login'))
        
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    """Get current logged-in user from session"""
    return session.get('user', None)

# Authentication routes
@app.route("/login")
def login():
    """Display login page"""
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template("login.html", firebase_available=firebase_available)

@app.route("/register")
def register():
    """Display registration page"""
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template("register.html", firebase_available=firebase_available)

@app.route("/dashboard")
@login_required
def dashboard():
    """Main dashboard - protected route"""
    user = get_current_user()
    return render_template("index_simple.html", user=user, firebase_available=firebase_available)

@app.route("/auth/login", methods=["POST"])
def auth_login():
    """Handle login form submission"""
    if not firebase_available:
        return jsonify({"success": False, "message": "Authentication system unavailable"})
    
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({"success": False, "message": "Email and password are required"})
        
        # Authenticate with Firebase
        user = auth.sign_in_with_email_and_password(email, password)
        
        # Store user info in session
        session['user'] = {
            'uid': user['localId'],
            'email': email,
            'token': user['idToken']
        }
        
        return jsonify({"success": True, "message": "Login successful", "redirect": url_for('dashboard')})
        
    except Exception as e:
        error_message = str(e)
        if "INVALID_EMAIL" in error_message:
            return jsonify({"success": False, "message": "Invalid email format"})
        elif "EMAIL_NOT_FOUND" in error_message:
            return jsonify({"success": False, "message": "Email not found"})
        elif "INVALID_PASSWORD" in error_message:
            return jsonify({"success": False, "message": "Invalid password"})
        else:
            return jsonify({"success": False, "message": "Login failed. Please try again."})

@app.route("/auth/register", methods=["POST"])
def auth_register():
    """Handle registration form submission"""
    if not firebase_available:
        return jsonify({"success": False, "message": "Authentication system unavailable"})
    
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        first_name = data.get('firstName', '')
        last_name = data.get('lastName', '')
        organization = data.get('organization', '')
        
        if not email or not password:
            return jsonify({"success": False, "message": "Email and password are required"})
        
        if len(password) < 6:
            return jsonify({"success": False, "message": "Password must be at least 6 characters"})
        
        # Create user with Firebase
        user = auth.create_user_with_email_and_password(email, password)
        
        # Store user info in session
        session['user'] = {
            'uid': user['localId'],
            'email': email,
            'firstName': first_name,
            'lastName': last_name,
            'organization': organization,
            'token': user['idToken']
        }
        
        return jsonify({"success": True, "message": "Registration successful", "redirect": url_for('dashboard')})
        
    except Exception as e:
        error_message = str(e)
        if "EMAIL_EXISTS" in error_message:
            return jsonify({"success": False, "message": "Email already exists"})
        elif "INVALID_EMAIL" in error_message:
            return jsonify({"success": False, "message": "Invalid email format"})
        elif "WEAK_PASSWORD" in error_message:
            return jsonify({"success": False, "message": "Password is too weak"})
        else:
            return jsonify({"success": False, "message": "Registration failed. Please try again."})

@app.route("/auth/logout", methods=["GET", "POST"])
def auth_logout():
    """Handle user logout"""
    session.pop('user', None)
    
    # Handle AJAX requests
    if request.method == "POST":
        return jsonify({"success": True, "message": "Logged out successfully", "redirect": url_for('login')})
    
    # Handle direct GET requests
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route("/")
def index():
    """Redirect to login or dashboard based on authentication"""
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route("/send_message", methods=["POST"])
@login_required
def send_message():
    """
    Enhanced chatbot endpoint with intelligent routing:
    1. Gets prompt from HTML
    2. Uses enhanced diagnostic processor with 70% confidence threshold
    3. Routes high-confidence queries: NLP codes + LLM analysis
    4. Routes low-confidence queries: Direct LLM consultation
    5. Returns appropriate response for chatbot display
    """
    try:
        # Get user message and image from the form
        user_message = request.form.get("message", "").strip()
        uploaded_file = request.files.get("image")
        
        # Validate input
        if not user_message and (not uploaded_file or not uploaded_file.filename):
            return jsonify({"reply": "Please provide a message or upload an image."})
        
        # Step 1: Enhanced NLP Processing with Intelligent Routing
        nlp_results = {}
        routing_decision = 'LLM_ONLY'  # Default fallback
        confidence_info = {}
        
        if user_message and nlp_available and diagnostic_processor:
            try:
                print(f"🔍 Processing with enhanced NLP routing: {user_message}")
                
                # Use enhanced processor with 90% confidence threshold
                nlp_results = diagnostic_processor.process_user_input(
                    user_message, 
                    top_n=5, 
                    confidence_threshold=90.0
                )
                
                routing_decision = nlp_results.get('analysis', {}).get('routing_decision', 'LLM_ONLY')
                route_to_llm_only = nlp_results.get('route_to_llm', True)
                
                confidence_info = {
                    'max_confidence': nlp_results.get('max_confidence', 0),
                    'high_confidence_count': nlp_results.get('high_confidence_count', 0),
                    'threshold': nlp_results.get('confidence_threshold', 70),
                    'diagnostic_relevance': nlp_results.get('analysis', {}).get('diagnostic_relevance', 0),
                    'relevance_class': nlp_results.get('analysis', {}).get('relevance_classification', 'unknown')
                }
                
                print(f"📊 Routing decision: {routing_decision}")
                print(f"📊 Max confidence: {confidence_info['max_confidence']:.1f}%")
                print(f"📊 High confidence matches: {confidence_info['high_confidence_count']}")
                print(f"📊 Diagnostic relevance: {confidence_info['diagnostic_relevance']:.1f}% ({confidence_info['relevance_class']})")
                
            except Exception as e:
                print(f"⚠️ Enhanced NLP processing error: {e}")
                routing_decision = 'LLM_ONLY'
                route_to_llm_only = True
        
        # Step 2: Build Gemini prompt based on routing decision
        if not llm_available:
            return jsonify({"reply": "❌ Gemini AI is not available. Please check your API key configuration."})
        
        # Create system prompt for vehicle expert
        system_prompt = """You are an expert vehicle diagnostics technician and automotive advisor. 
        Provide clear, practical, and safety-focused advice. Use simple language that car owners can understand.
        When diagnostic codes are provided, explain them thoroughly and give step-by-step troubleshooting advice.
        If there are clarity issues in the user's description, ask clarifying questions.
        Provide step-by-step solutions, safety tips, and maintenance advice.
        Always prioritize safety and recommend professional help when necessary.
        """
        
        # Build enhanced prompt based on routing decision
        if routing_decision == 'NLP' and nlp_results.get('matches'):
            # High-confidence route: Include NLP diagnostic codes
            enhanced_prompt = _build_nlp_enhanced_prompt(user_message, nlp_results, confidence_info)
            print("🎯 Using NLP-to-LLM routing with diagnostic codes")
        else:
            # Low-confidence route: Direct LLM consultation
            enhanced_prompt = _build_direct_llm_prompt(user_message, nlp_results, confidence_info)
            print("🤖 Using direct LLM routing for general consultation")
        
        # Prepare content for Gemini
        contents = []
        
        # Step 3: Handle image if provided
        if uploaded_file and uploaded_file.filename != '':
            try:
                # Read and encode image
                image_data = uploaded_file.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                # Create image content for Gemini
                image_content = {
                    "inline_data": {
                        "mime_type": uploaded_file.content_type or "image/jpeg",
                        "data": image_base64
                    }
                }
                
                # Add image first (Gemini works better this way)
                contents.append(image_content)
                
                # Add image analysis instruction
                if routing_decision == 'NLP':
                    enhanced_prompt += "\n🖼️ IMAGE ANALYSIS: Please also analyze the uploaded image in relation to the diagnostic codes above. Look for visual confirmation of the identified issues."
                else:
                    enhanced_prompt += "\n🖼️ IMAGE ANALYSIS: Please analyze this vehicle image and provide relevant automotive advice based on what you see."
                    
            except Exception as e:
                return jsonify({"reply": f"❌ Error processing image: {str(e)}"})
        
        # Add the complete prompt
        final_prompt = system_prompt + "\n\n" + enhanced_prompt
        contents.append(final_prompt)
        
        # Step 4: Send to Gemini and get response
        try:
            print("🤖 Sending enhanced request to Gemini AI...")
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(contents)
            
            gemini_response = response.text
            
            # Add routing information to response for user awareness
            routing_info = _generate_routing_info(routing_decision, confidence_info)
            final_response = f"{gemini_response}\n\n{routing_info}"
            
            print("✅ Received response from Gemini AI")
            
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            final_response = f"Sorry, I encountered an error while processing your request with Gemini AI: {str(e)}"
        
        # Step 4: Search for related YouTube videos (if OBD codes found)
        video_results = None
        if is_youtube_available():
            try:
                # Extract OBD codes and symptoms for video search
                obd_codes = []
                symptoms = []
                
                if nlp_results and nlp_results.get('matches'):
                    # Get high-confidence OBD codes
                    for match in nlp_results.get('matches', []):
                        confidence = match.get('confidence_score', 0)
                        if isinstance(confidence, str) and '%' in confidence:
                            confidence = float(confidence.strip('%'))
                        
                        if confidence >= 70:  # Only high-confidence codes
                            obd_codes.append(match.get('code_id', ''))
                
                # Get detected symptoms
                analysis = nlp_results.get('analysis', {})
                symptoms = analysis.get('detected_symptoms', [])
                
                # Search for videos if we have relevant data
                if obd_codes or symptoms:
                    print(f"🎥 Searching YouTube videos for codes: {obd_codes}, symptoms: {symptoms}")
                    video_results = search_diagnostic_videos(
                        obd_codes=obd_codes,
                        symptoms=symptoms,
                        user_prompt=user_message,
                        max_results=3
                    )
                    
                    if video_results and video_results.get('has_videos'):
                        print(f"✅ Found {video_results['count']} relevant videos")
                    else:
                        print("ℹ️ No relevant videos found")
                        
            except Exception as video_error:
                print(f"⚠️ YouTube video search error: {video_error}")
                video_results = None
        
        # Step 5: Build complete response with videos
        response_data = {"reply": final_response}
        
        if video_results and video_results.get('has_videos'):
            response_data["videos"] = video_results
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ General error in send_message: {e}")
        return jsonify({"reply": f"Sorry, an unexpected error occurred: {str(e)}"})

def _build_nlp_enhanced_prompt(user_message, nlp_results, confidence_info):
    """Build clean prompt for high-confidence NLP-to-LLM routing without exposing confidence details"""
    enhanced_prompt = f"User Question: {user_message}\n\n"
    enhanced_prompt += "🎯 DIAGNOSTIC ANALYSIS:\n\n"
    
    matches = nlp_results.get('matches', [])
    analysis = nlp_results.get('analysis', {})
    
    for i, match in enumerate(matches, 1):
        enhanced_prompt += f"{i}. **OBD Code: {match.get('code_id', 'Unknown')}**\n"
        enhanced_prompt += f"   Description: {match.get('description', 'No description')}\n"
        enhanced_prompt += f"   Priority: {match.get('priority', 'Medium')}\n"
        
        if match.get('common_causes'):
            enhanced_prompt += "   Common Causes:\n"
            for cause in match['common_causes']:
                enhanced_prompt += f"   • {cause}\n"
        
        if match.get('confirmation'):
            enhanced_prompt += f"   Most Likely Cause: {match['confirmation']}\n"
        
        enhanced_prompt += "\n"
    
    # Add analysis context without confidence details
    if analysis.get('detected_symptoms'):
        enhanced_prompt += f"Detected Symptoms: {', '.join(analysis['detected_symptoms'])}\n\n"
    
    enhanced_prompt += """Please provide a comprehensive expert analysis that:
1. Confirms and explains these diagnostic codes in detail
2. Provides step-by-step troubleshooting and repair guidance
3. Explains the relationship between codes if multiple are present
4. Gives specific safety precautions and when to seek professional help
5. Suggests preventive maintenance to avoid recurrence
6. Estimates repair complexity and potential costs if appropriate

"""
    return enhanced_prompt

def _build_direct_llm_prompt(user_message, nlp_results, confidence_info):
    """Build clean prompt for low-confidence direct LLM routing"""
    enhanced_prompt = f"User Question: {user_message}\n\n"
    
    # Explain why direct routing was chosen without exposing confidence details
    reason = nlp_results.get('reason', 'Query requires general automotive consultation')
    enhanced_prompt += f"🤖 GENERAL CONSULTATION MODE: {reason}\n\n"
    
    # Include diagnostic hints if available, but without confidence details
    if nlp_results.get('matches'):
        enhanced_prompt += "📋 DIAGNOSTIC HINTS:\n"
        enhanced_prompt += f"Found {len(nlp_results['matches'])} potential diagnostic codes:\n"
        
        for i, match in enumerate(nlp_results['matches'][:3], 1):  # Limit to top 3
            enhanced_prompt += f"{i}. {match.get('code_id', 'Unknown')}: {match.get('description', 'No description')}\n"
        
        enhanced_prompt += "\nNote: These codes are provided as potential hints only.\n\n"
    
    enhanced_prompt += """Please provide helpful automotive guidance that:
1. Addresses the user's question with general automotive knowledge
2. Asks clarifying questions if the issue description is unclear
3. Provides practical troubleshooting steps appropriate for the question
4. Suggests when professional diagnosis might be needed
5. Gives relevant safety advice and maintenance tips
6. Uses any diagnostic hints carefully and with appropriate disclaimers

"""
    return enhanced_prompt

def _generate_routing_info(routing_decision, confidence_info):
    """Generate clean user-friendly routing information without confidence details"""
    if routing_decision == 'NLP':
        return f"🎯 *Analysis based on diagnostic code matches*"
    else:
        return "🤖 *General automotive consultation*"

@app.route("/diagnose", methods=["POST"])
@login_required
def diagnose():
    """Enhanced NLP diagnosis endpoint with routing information"""
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()
        confidence_threshold = data.get("confidence_threshold", 50.0)
        
        if not user_input:
            return jsonify({
                "error": "Please provide a description of your vehicle problem",
                "success": False
            })
        
        if not nlp_available or diagnostic_processor is None:
            return jsonify({
                "error": "NLP diagnostic system is not available",
                "success": False
            })
        
        # Process the user input with enhanced NLP
        results = diagnostic_processor.process_user_input(
            user_input, 
            top_n=5, 
            confidence_threshold=confidence_threshold
        )
        
        # Handle different routing scenarios
        if not results.get('success', True):
            return jsonify({
                "success": False,
                "route_to_llm_only": results.get('route_to_llm', True),
                "routing_decision": results.get('routing_decision', 'LLM_ONLY'),
                "reason": results.get('reason', 'Processing failed'),
                "diagnostic_relevance": results.get('diagnostic_relevance', 0),
                "relevance_classification": results.get('relevance_classification', 'unknown'),
                "matches": [],
                "analysis": results.get('analysis', {})
            })
        
        # Format the response for enhanced routing
        formatted_matches = []
        for match in results.get('matches', []):
            formatted_match = {
                "code": match.get('code_id', 'Unknown'),
                "description": match.get('description', 'No description'),
                "priority": match.get('priority', 'Medium'),
                "confidence": match.get('confidence_percentage', 'Unknown'),
                "confidence_score": match.get('confidence_score', 0),
                "confidence_breakdown": match.get('confidence_breakdown', {}),
                "common_causes": match.get('common_causes', []),
                "likely_cause": match.get('confirmation', ''),
                "source_method": match.get('source_method', 'Unknown')
            }
            formatted_matches.append(formatted_match)
        
        # Enhanced response with routing information
        response_data = {
            "success": True,
            "route_to_llm_only": results.get('route_to_llm', False),
            "routing_decision": "NLP_TO_LLM" if results.get('analysis', {}).get('routing_decision') == 'NLP' else "LLM_ONLY",
            "message": results.get('message', f"Found {len(formatted_matches)} diagnostic matches"),
            "user_input": user_input,
            "matches": formatted_matches,
            "confidence_analysis": {
                "max_confidence": results.get('max_confidence', 0),
                "high_confidence_count": results.get('high_confidence_count', 0),
                "confidence_threshold": results.get('confidence_threshold', confidence_threshold),
                "diagnostic_relevance": results.get('diagnostic_relevance', 0),
                "relevance_classification": results.get('relevance_classification', 'unknown')
            },
            "analysis": {
                "detected_symptoms": results.get('analysis', {}).get('detected_symptoms', []),
                "total_matches": len(formatted_matches),
                "enhanced_matches": results.get('analysis', {}).get('enhanced_matches', 0),
                "routing_decision": "NLP_TO_LLM" if results.get('analysis', {}).get('routing_decision') == 'NLP' else "LLM_ONLY",
                "processing_method": "enhanced_confidence_routing"
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in enhanced diagnose endpoint: {e}")
        return jsonify({
            "error": f"An error occurred during diagnosis: {str(e)}",
            "success": False,
            "route_to_llm_only": True,
            "routing_decision": "LLM_ONLY"
        })

@app.route("/transcribe_audio", methods=["POST"])
def transcribe_audio():
    """Speech-to-text endpoint using OpenAI Whisper"""
    try:
        if 'audio' not in request.files:
            return jsonify({
                "error": "No audio file provided", 
                "success": False
            }), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({
                "error": "No audio file selected", 
                "success": False
            }), 400
        
        # Validate audio file
        validator = AudioValidator()
        is_valid, validation_error = validator.validate_file(audio_file)
        
        if not is_valid:
            return jsonify({
                "error": f"Invalid audio file: {validation_error}", 
                "success": False
            }), 400
        
        # Save temporary file for processing
        processor = AudioProcessor()
        temp_path = None
        
        try:
            temp_path = processor.save_temp_file(audio_file)
            
            # Transcribe using Whisper
            result = transcribe_audio_file(temp_path)
            
            if result['success']:
                logger.info(f"Successfully transcribed audio: {result['text'][:50]}...")
                return jsonify({
                    "success": True,
                    "text": result['text'],
                    "confidence": result.get('confidence', 0.0),
                    "duration": result.get('duration', 0.0)
                })
            else:
                logger.error(f"Transcription failed: {result['error']}")
                return jsonify({
                    "error": result['error'], 
                    "success": False
                }), 500
                
        finally:
            # Clean up temporary file
            if temp_path:
                processor.cleanup_temp_file(temp_path)
                
    except Exception as e:
        logger.error(f"Error in transcribe_audio endpoint: {e}")
        return jsonify({
            "error": f"An error occurred during transcription: {str(e)}",
            "success": False
        }), 500

@app.route("/speech_status", methods=["GET"])
def speech_status():
    """Check speech recognition system status"""
    try:
        status = get_speech_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting speech status: {e}")
        return jsonify({
            "error": f"Failed to get speech status: {str(e)}",
            "success": False,
            "whisper_available": False
        }), 500

@app.route("/search_youtube", methods=["POST"])
def search_youtube():
    """Manual YouTube video search endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
            
        if not is_youtube_available():
            return jsonify({"error": "YouTube search is not available"}), 503
        
        # Perform manual search
        video_results = search_manual_videos(query, max_results=5)
        
        if video_results and video_results.get('has_videos'):
            return jsonify({
                "success": True,
                "videos": video_results
            })
        else:
            return jsonify({
                "success": False,
                "message": "No videos found for your search query"
            })
            
    except Exception as e:
        print(f"❌ YouTube search error: {e}")
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

@app.route("/youtube_status", methods=["GET"])
def youtube_status():
    """Check YouTube search availability"""
    return jsonify({
        "available": is_youtube_available(),
        "message": "YouTube search is ready" if is_youtube_available() else "YouTube API not configured"
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint to verify system status"""
    return jsonify({
        "nlp_available": nlp_available,
        "llm_available": llm_available,
        "youtube_available": is_youtube_available(),
        "status": "healthy" if (nlp_available and llm_available) else "degraded",
        "diagnostic_processor": "available" if diagnostic_processor else "unavailable",
        "gemini_api": "configured" if llm_available else "not configured"
    })

@app.route("/ml-diagnostics")
def ml_diagnostics():
    """ML Diagnostics page for vehicle engine anomaly detection"""
    return render_template("ml_diagnostics.html")

@app.route("/api/ml-diagnose", methods=["POST"])
def ml_diagnose():
    """API endpoint for ML-based vehicle diagnostics"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Predict anomaly using rule-based system
        result = predict_anomaly_rule_based(data)
        
        if "error" in result:
            return jsonify(result), 400
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ml-ranges", methods=["GET"])
def get_ml_normal_ranges():
    """Get normal operating ranges for ML diagnostics reference"""
    return jsonify({
        "normal_ranges": NORMAL_RANGES,
        "features": FEATURES
    })

if __name__ == "__main__":
    print("🚗 Starting Vehicle Diagnostics AI Chatbot...")
    print(f"📊 NLP Diagnosis: {'✅ Available' if nlp_available else '❌ Not Available'}")
    print(f"🤖 Gemini AI: {'✅ Available' if llm_available else '❌ Not Available'}")
    print("🌐 Access at: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)