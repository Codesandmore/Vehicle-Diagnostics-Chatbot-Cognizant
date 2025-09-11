
import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify

# Configure the Gemini API key
# Make sure to set the GOOGLE_API_KEY environment variable in your environment.
# For example, in your terminal: export GOOGLE_API_KEY="YOUR_API_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Create a Gemini Pro model
model = genai.GenerativeModel('gemini-pro')

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/send_message", methods=["POST"])
def send_message():
    user_message = request.json["message"]
    
    # Send the user's message to the Gemini model
    response = model.generate_content(user_message)
    
    # Extract the bot's reply from the response
    bot_reply = response.text
    
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
