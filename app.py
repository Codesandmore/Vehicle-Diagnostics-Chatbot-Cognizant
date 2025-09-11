
import os
from google import genai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

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
    user_message = request.json["message"]
    
    # Send the user's message to the Gemini model using the new API
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[user_message]
    )
    
    # Extract the bot's reply from the response
    bot_reply = response.text
    
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
