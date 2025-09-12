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
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    llm_available = True
else:
    print("Warning: GOOGLE_API_KEY not found in environment variables")
    llm_available = False

app = Flask(__name__)

# Initialize the diagnostic processor for NLP functionality
try:
    obd_codes_path = os.path.join(os.path.dirname(__file__), 'data', 'obd_codes.json')
    diagnostic_processor = DiagnosticProcessor(obd_codes_path)
    print("✅ Diagnostic processor initialized successfully")
    nlp_available = True
except Exception as e:
    print(f"❌ Error initializing diagnostic processor: {e}")
    diagnostic_processor = None
    nlp_available = False

@app.route("/")
def index():
    return render_template("index_simple.html")

@app.route("/send_message", methods=["POST"])
def send_message():
    """
    Main chatbot endpoint that:
    1. Gets prompt from HTML
    2. Processes with diagnostic processor to get OBD codes
    3. Combines prompt + OBD codes and sends to Gemini
    4. Returns Gemini's response for chatbot display
    """
    try:
        # Get user message and image from the form
        user_message = request.form.get("message", "").strip()
        uploaded_file = request.files.get("image")
        
        # Validate input
        if not user_message and (not uploaded_file or not uploaded_file.filename):
            return jsonify({"reply": "Please provide a message or upload an image."})
        
        # Step 1: Process user message with NLP diagnostic processor to get OBD codes
        obd_codes_info = []
        nlp_analysis = {}
        
        if user_message and nlp_available and diagnostic_processor:
            try:
                print(f"🔍 Processing message with NLP: {user_message}")
                nlp_results = diagnostic_processor.process_user_input(user_message, top_n=5)
                nlp_analysis = nlp_results.get('analysis', {})
                
                # Extract OBD codes with their details
                for match in nlp_results.get('matches', []):
                    confidence_score = match.get('confidence_score', 0)
                    
                    # Convert percentage string to float if needed
                    if isinstance(confidence_score, str) and '%' in confidence_score:
                        confidence_score = float(confidence_score.strip('%')) / 100
                    
                    obd_code_info = {
                        'code': match.get('code_id', ''),
                        'description': match.get('description', ''),
                        'confidence': confidence_score,
                        'priority': match.get('priority', 'Medium'),
                        'common_causes': match.get('common_causes', []),
                        'likely_cause': match.get('confirmation', '')
                    }
                    obd_codes_info.append(obd_code_info)
                    
                print(f"📋 Found {len(obd_codes_info)} OBD code matches")
                
            except Exception as e:
                print(f"⚠️ NLP processing error: {e}")
        
        # Step 2: Build enhanced prompt for Gemini with OBD codes
        if not llm_available:
            return jsonify({"reply": "❌ Gemini AI is not available. Please check your API key configuration."})
        
        # Create system prompt for vehicle expert
        system_prompt = """You are an expert vehicle diagnostics technician and automotive advisor. 
        Provide clear, practical, and safety-focused advice. Use simple language that car owners can understand.
        When diagnostic codes are provided, explain them thoroughly and give step-by-step troubleshooting advice."""
        
        # Build the enhanced prompt with OBD codes
        enhanced_prompt = f"User Question: {user_message}\n\n"
        
        # Add OBD codes information if found
        if obd_codes_info:
            enhanced_prompt += "🔍 DIAGNOSTIC ANALYSIS RESULTS:\n\n"
            
            for i, code_info in enumerate(obd_codes_info, 1):
                enhanced_prompt += f"{i}. **OBD Code: {code_info['code']}**\n"
                enhanced_prompt += f"   Description: {code_info['description']}\n"
                enhanced_prompt += f"   Confidence: {code_info['confidence']:.1%}\n"
                enhanced_prompt += f"   Priority: {code_info['priority']}\n"
                
                if code_info['common_causes']:
                    enhanced_prompt += f"   Common Causes:\n"
                    for cause in code_info['common_causes']:
                        enhanced_prompt += f"   • {cause}\n"
                
                if code_info['likely_cause']:
                    enhanced_prompt += f"   Most Likely Cause: {code_info['likely_cause']}\n"
                
                enhanced_prompt += "\n"
            
            # Add analysis context
            if nlp_analysis.get('detected_symptoms'):
                enhanced_prompt += f"Detected Symptoms: {', '.join(nlp_analysis['detected_symptoms'])}\n\n"
            
            enhanced_prompt += """Please provide a comprehensive response that:
1. Explains what these diagnostic codes mean in simple terms
2. Confirms or provides additional insights about the diagnosis
3. Gives step-by-step troubleshooting instructions
4. Mentions any safety precautions
5. Suggests when to seek professional help
6. Provides preventive maintenance tips

"""
        else:
            enhanced_prompt += "No specific diagnostic codes were identified, but please provide helpful vehicle advice based on the user's question.\n\n"
        
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
                if obd_codes_info:
                    enhanced_prompt += "🖼️ IMAGE ANALYSIS: Please also analyze the uploaded image in relation to the diagnostic codes above. Look for visual confirmation of the identified issues.\n"
                else:
                    enhanced_prompt += "🖼️ IMAGE ANALYSIS: Please analyze this vehicle image and identify any visible issues or provide relevant advice.\n"
                    
            except Exception as e:
                return jsonify({"reply": f"❌ Error processing image: {str(e)}"})
        
        # Add the complete prompt
        final_prompt = system_prompt + "\n\n" + enhanced_prompt
        contents.append(final_prompt)
        
        # Step 4: Send to Gemini and get response
        try:
            print("🤖 Sending request to Gemini AI...")
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(contents)
            
            gemini_response = response.text
            print("✅ Received response from Gemini AI")
            
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            gemini_response = f"Sorry, I encountered an error while processing your request with Gemini AI: {str(e)}"
        
        # Step 5: Return response for chatbot display
        return jsonify({"reply": gemini_response})
        
    except Exception as e:
        print(f"❌ General error in send_message: {e}")
        return jsonify({"reply": f"Sorry, an unexpected error occurred: {str(e)}"})

@app.route("/diagnose", methods=["POST"])
def diagnose():
    """Pure NLP diagnosis endpoint for testing"""
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
        formatted_matches = []
        for match in results['matches']:
            formatted_match = {
                "code": match['code_id'],
                "description": match['description'],
                "priority": match['priority'],
                "confidence": f"{match['confidence_score']:.1%}",
                "common_causes": match['common_causes'],
                "likely_cause": match['confirmation']
            }
            formatted_matches.append(formatted_match)
        
        response_data = {
            "success": True,
            "user_input": user_input,
            "matches": formatted_matches,
            "analysis": {
                "detected_symptoms": results['analysis'].get('detected_symptoms', []),
                "total_matches": len(formatted_matches)
            },
            "message": f"Found {len(formatted_matches)} potential diagnostic code(s) based on your description."
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in diagnose endpoint: {e}")
        return jsonify({
            "error": f"An error occurred during diagnosis: {str(e)}",
            "success": False
        })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint to verify system status"""
    return jsonify({
        "nlp_available": nlp_available,
        "llm_available": llm_available,
        "status": "healthy" if (nlp_available and llm_available) else "degraded",
        "diagnostic_processor": "available" if diagnostic_processor else "unavailable",
        "gemini_api": "configured" if llm_available else "not configured"
    })

if __name__ == "__main__":
    print("🚗 Starting Vehicle Diagnostics AI Chatbot...")
    print(f"📊 NLP Diagnosis: {'✅ Available' if nlp_available else '❌ Not Available'}")
    print(f"🤖 Gemini AI: {'✅ Available' if llm_available else '❌ Not Available'}")
    print("🌐 Access at: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)