
import os
import base64
from google import genai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
# Make sure to set the GOOGLE_API_KEY environment variable in your environment.
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Create a Gemini client
client = genai.Client()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/send_message", methods=["POST"])
def send_message():
    user_message = request.form.get("message", "")
    uploaded_file = request.files.get("image")
    
    contents = []
    
    # Add text message if provided
    if user_message:
        contents.append(user_message)
    
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
                contents.append("Please analyze this vehicle image and provide diagnostic insights.")
                
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
        bot_reply = f"Sorry, I encountered an error: {str(e)}"
    
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
