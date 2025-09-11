# Vehicle Diagnostics Chatbot

A Flask-based chatbot application that uses Google's Gemini AI for vehicle diagnostics. The chatbot supports both text-based conversations and image analysis for vehicle-related questions.

## Features

- **Text-based chat**: Ask questions about vehicle diagnostics and maintenance
- **Image upload**: Upload vehicle images for AI-powered visual analysis
- **Real-time responses**: Get instant responses from Google's Gemini AI
- **Clean UI**: Modern, responsive chat interface
- **Environment security**: API keys stored safely in `.env` files

## Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Vehicle-Diagnostics-Chatbot-Cognizant
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   - Copy `.env` file and add your Google Gemini API key:

   ```
   GOOGLE_API_KEY="your_actual_api_key_here"
   ```

5. **Run the application**

   ```bash
   python app.py
   ```

6. **Open your browser**
   - Navigate to `http://127.0.0.1:5000`

## Usage

### Text Chat

- Type your vehicle-related questions in the input field
- Press Enter or click Send to get AI responses

### Image Upload

- Click the camera button (📷) to upload an image
- Select an image file from your device
- The image will be previewed before sending
- Add optional text description or send image alone
- AI will analyze the image and provide diagnostic insights

## Supported Image Formats

- JPEG, PNG, GIF, BMP, and other common image formats
- Maximum recommended size: 10MB

## Technology Stack

- **Backend**: Flask (Python)
- **AI**: Google Gemini 2.0 Flash
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: Pillow (PIL)
- **Environment Management**: python-dotenv

## Project Structure

```
Vehicle-Diagnostics-Chatbot-Cognizant/
├── app.py                 # Main Flask application
├── .env                   # Environment variables (API key)
├── .gitignore            # Git ignore file
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── templates/
│   └── index.html        # Frontend HTML template
└── static/
    ├── style.css         # CSS styling
    └── script.js         # JavaScript functionality
```

## Security Notes

- API keys are stored in `.env` file (excluded from git)
- Temporary image files are automatically cleaned up
- File upload validation prevents malicious uploads

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request
