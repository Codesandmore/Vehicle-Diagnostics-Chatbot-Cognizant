import os
import base64
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from nlp.diagnostic_processor import DiagnosticProcessor
from PIL import Image
import io
import json

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
# Make sure to set the GOOGLE_API_KEY environment variable in your environment.
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Create a Gemini client (only if API key is available)
try:
    if api_key:
        client = genai.Client()
        llm_available = True
    else:
        print("Warning: GOOGLE_API_KEY not found in environment variables")
        client = None
        llm_available = False
except Exception as e:
    print(f"Warning: LLM functionality unavailable - {e}")
    client = None
    llm_available = False

app = Flask(__name__)

# Initialize the diagnostic processor for NLP functionality
try:
    obd_codes_path = os.path.join(os.path.dirname(__file__), 'data', 'obd_codes.json')
    diagnostic_processor = DiagnosticProcessor(obd_codes_path)
    print("Diagnostic processor initialized successfully")
    nlp_available = True
except Exception as e:
    print(f"Error initializing diagnostic processor: {e}")
    diagnostic_processor = None
    nlp_available = False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/diagnose", methods=["POST"])
def diagnose():
    """NLP-based diagnostic endpoint"""
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()
        
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
        
        # Process the user input with NLP
        results = diagnostic_processor.process_user_input(user_input, top_n=5)
        
        if not results.get('matches'):
            return jsonify({
                "message": "No matching diagnostic codes found. Please provide more specific details about your vehicle problem.",
                "success": True,
                "matches": [],
                "analysis": results.get('analysis', {})
            })
        
        # Format the response
        matches = results['matches']
        analysis = results.get('analysis', {})
        
        # Add confidence scores and format matches
        formatted_matches = []
        for match in matches:
            formatted_match = {
                'code': match['code_id'],
                'description': match['description'],
                'common_causes': match.get('common_causes', []),
                'priority': match.get('priority', 'Medium'),
                'confidence': f"{match.get('confidence_score', 0.5) * 100:.1f}%",
                'confirmation': match.get('confirmation', '')
            }
            formatted_matches.append(formatted_match)
        
        # Create response message
        if len(formatted_matches) == 1:
            message = "Found 1 matching diagnostic code:"
        else:
            message = f"Found {len(formatted_matches)} potential diagnostic codes:"
        
        return jsonify({
            "success": True,
            "message": message,
            "matches": formatted_matches,
            "analysis": analysis
        })
        
    except Exception as e:
        print(f"Error in diagnose endpoint: {e}")
        return jsonify({
            "error": f"An error occurred during diagnosis: {str(e)}",
            "success": False
        })

@app.route("/send_message", methods=["POST"])
def send_message():
    """LLM-based chat endpoint with image support"""
    if not llm_available:
        return jsonify({"reply": "LLM functionality is not available. Please check your Google API key configuration."})
    
    user_message = request.form.get("message", "")
    uploaded_file = request.files.get("image")
    
    contents = []
    
    # Add text message if provided
    if user_message:
        # Add vehicle diagnostic context to the message
        vehicle_context = """You are a vehicle diagnostic AI assistant. When answering questions about vehicles, 
        provide practical, safety-focused advice. If asked to analyze images, look for diagnostic clues like 
        warning lights, visible damage, fluid leaks, or component issues."""
        contents.append(vehicle_context + "\n\nUser question: " + user_message)
    
    # Handle image upload if provided
    if uploaded_file and uploaded_file.filename != '':
        try:
            # Read the image file
            image_data = uploaded_file.read()
            
            # Convert image to base64 for inline processing
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Create image content for Gemini
            image_content = {
                "inline_data": {
                    "mime_type": uploaded_file.content_type or "image/jpeg",
                    "data": image_base64
                }
            }
            contents.append(image_content)
            
            # If no text message provided, add a default prompt for image analysis
            if not user_message:
                contents.append("Please analyze this vehicle-related image and provide diagnostic insights, maintenance advice, or identify any issues you can see.")
                
        except Exception as e:
            return jsonify({"reply": f"Error processing image: {str(e)}"})
    
    # If no content provided, return error
    if not contents:
        return jsonify({"reply": "Please provide a message or upload an image."})
    
    try:
        # Send the content to the Gemini model
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents
        )
        
        # Extract the bot's reply from the response
        bot_reply = response.text
        
    except Exception as e:
        bot_reply = f"Sorry, I encountered an error while processing your request: {str(e)}"
    
    return jsonify({"reply": bot_reply})

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint to verify system status"""
    return jsonify({
        "nlp_available": nlp_available,
        "llm_available": llm_available,
        "status": "healthy" if (nlp_available or llm_available) else "degraded"
    })

if __name__ == "__main__":
    print(f"Starting Vehicle Diagnostics AI Chatbot...")
    print(f"NLP Diagnosis: {'Available' if nlp_available else 'Not Available'}")
    print(f"LLM Chat: {'Available' if llm_available else 'Not Available'}")
    app.run(debug=True)