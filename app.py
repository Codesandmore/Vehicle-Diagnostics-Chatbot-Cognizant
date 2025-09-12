
import os
import base64
import json
from google import genai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from PIL import Image
import io
from nlp.diagnostic_processor import DiagnosticProcessor

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
# Make sure to set the GOOGLE_API_KEY environment variable in your environment.
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Create a Gemini client
client = genai.Client()

# Initialize the diagnostic processor
try:
    obd_codes_path = os.path.join(os.path.dirname(__file__), 'data', 'obd_codes.json')
    diagnostic_processor = DiagnosticProcessor(obd_codes_path)
    print("Diagnostic processor initialized successfully")
except Exception as e:
    print(f"Error initializing diagnostic processor: {e}")
    diagnostic_processor = None

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/diagnose", methods=["POST"])
def diagnose():
    """Main endpoint for vehicle diagnostics"""
    try:
        if request.method == "POST":
            if request.is_json:
                data = request.get_json()
                user_input = data.get("message", "").strip()
            else:
                user_input = request.form.get("message", "").strip()
        else:
            user_input = request.args.get("message", "").strip()
        
        if not user_input:
            return jsonify({
                "error": "Please provide a description of your vehicle problem",
                "success": False
            })
               
        if diagnostic_processor is None:
            return jsonify({
                "error": "Diagnostic system is not available",
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

@app.route("/send_message", methods=["POST"])
def send_message():
    """
    Enhanced endpoint that processes user input with NLP diagnostics first,
    then passes high-confidence OBD codes to Gemini AI along with the prompt
    """
    user_message = request.form.get("message", "").strip()
    uploaded_file = request.files.get("image")
    
    # Check if we have any input
    if not user_message and (not uploaded_file or not uploaded_file.filename):
        return jsonify({"reply": "Please provide a message or upload an image."})
    
    # Build enhanced prompt: use local NLP first and include high-confidence matches
    prompt_parts = []
    high_conf_matches = []
    
    # Process user input with NLP if a message is provided
    if user_message:
        prompt_parts.append(f"User question: {user_message}")
        
        # Try to get diagnostic codes using the NLP processor
        try:
            if diagnostic_processor:
                # Process user input and get diagnostic results
                nlp_results = diagnostic_processor.process_user_input(user_message, top_n=5)
                
                # Filter for high confidence matches (>=90%)
                for match in nlp_results.get('matches', []):
                    # Convert confidence score to float if it's a string percentage
                    confidence_score = match.get('confidence_score', 0)
                    if isinstance(confidence_score, str) and '%' in confidence_score:
                        confidence_score = float(confidence_score.strip('%')) / 100
                    
                    # Add high confidence matches to our list
                    if confidence_score >= 0.9:
                        high_conf_matches.append({
                            'code': match.get('code_id'),
                            'description': match.get('description'),
                            'confidence': confidence_score,
                            'common_causes': match.get('common_causes', []),
                            'likely_cause': match.get('confirmation', '')
                        })
        except Exception as e:
            print(f"NLP processor error: {e}")
    
    # Prepare content list for Gemini
    contents = []
    
    # If we have high confidence matches, add them to the prompt
    if high_conf_matches:
        prompt_parts.append("\nHIGH CONFIDENCE OBD CODE MATCHES (>=90%):")
        for match in high_conf_matches:
            prompt_parts.append(f"- Code {match['code']}: {match['description']} (confidence: {match['confidence']:.1%})")
            if match['common_causes']:
                prompt_parts.append(f"  Common causes: {', '.join(match['common_causes'])}")
            if match['likely_cause']:
                prompt_parts.append(f"  Likely cause: {match['likely_cause']}")
        
        # Add instruction for Gemini to use the code information
        prompt_parts.append("\nPlease incorporate these diagnostic codes in your analysis. Explain what they mean and provide relevant repair advice based on these codes.")
    
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
            
            # Add image as first content (Gemini works better with image first)
            contents.append(image_content)
            
            # Add image-specific instructions
            if high_conf_matches:
                prompt_parts.append("\nAlso analyze this image in relation to the identified diagnostic codes.")
            else:
                prompt_parts.append("\nPlease analyze this vehicle image and provide diagnostic insights.")
                
        except Exception as e:
            return jsonify({"reply": f"Error processing image: {str(e)}"})
    
    # Create final prompt with vehicle expert context
    vehicle_context = """
You are a vehicle diagnostics expert. Focus on providing:
1. Clear explanation of vehicle problems
2. Specific diagnostic details and error codes
3. Practical repair advice
4. Safety warnings when appropriate
5. Format your response with markdown for readability
"""
    final_prompt = vehicle_context + "\n\n" + "\n".join(prompt_parts)
    
    # Add the text prompt as the last content item
    contents.append(final_prompt)
    
    try:
        # Send the content to the Gemini model
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents
        )
        
        # Extract the bot's reply from the response
        bot_reply = response.text
        
    except Exception as e:
        bot_reply = f"Sorry, I encountered an error: {str(e)}"
    
    return jsonify({"reply": bot_reply})

@app.route("/get_code", methods=["POST"])
def get_code():
    """Get detailed information about a specific OBD code"""
    try:
        data = request.get_json()
        code_id = data.get("code_id", "").strip().upper()
        
        if not code_id:
            return jsonify({
                "error": "Please provide a valid OBD code ID",
                "success": False
            })
        
        if diagnostic_processor is None:
            return jsonify({
                "error": "Diagnostic system is not available",
                "success": False
            })
        
        # Get code details
        result = diagnostic_processor.get_code_details(code_id)
        
        if result['found']:
            code_data = result['code']
            return jsonify({
                "success": True,
                "code": {
                    "id": code_data['id'],
                    "description": code_data['description'],
                    "common_causes": code_data.get('common_causes', []),
                    "priority": code_data.get('priority', 'Unknown'),
                    "likely_cause": code_data.get('confirmation', 'Not specified')
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": result['message']
            })
            
    except Exception as e:
        print(f"Error in get_code endpoint: {e}")
        return jsonify({
            "error": f"An error occurred: {str(e)}",
            "success": False
        })

@app.route("/get_codes_by_priority", methods=["POST"])
def get_codes_by_priority():
    """Get all codes filtered by priority level"""
    try:
        data = request.get_json()
        priority = data.get("priority", "").strip()
        
        if priority.lower() not in ['high', 'medium', 'low']:
            return jsonify({
                "error": "Priority must be 'high', 'medium', or 'low'",
                "success": False
            })
        
        if diagnostic_processor is None:
            return jsonify({
                "error": "Diagnostic system is not available",
                "success": False
            })
        
        # Get codes by priority
        result = diagnostic_processor.get_all_codes_by_priority(priority)
        
        return jsonify({
            "success": True,
            "priority": result['priority'],
            "count": result['count'],
            "codes": result['codes']
        })
        
    except Exception as e:
        print(f"Error in get_codes_by_priority endpoint: {e}")
        return jsonify({
            "error": f"An error occurred: {str(e)}",
            "success": False
        })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    processor_status = "available" if diagnostic_processor is not None else "unavailable"
    return jsonify({
        "status": "healthy",
        "diagnostic_processor": processor_status,
        "total_codes": len(diagnostic_processor.obd_codes) if diagnostic_processor else 0
    })

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
