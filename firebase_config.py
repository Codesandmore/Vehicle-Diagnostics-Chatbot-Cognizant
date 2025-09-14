# Firebase Configuration for Vehicle Diagnostics Chatbot
# This is a PYTHON file, not JavaScript!
# 
# SETUP INSTRUCTIONS:
# The configuration below has been converted from your Firebase JavaScript config
# to the correct Python format for use with pyrebase4

FIREBASE_CONFIG = {
    "apiKey": "AIzaSyDQ2rq5FqugU20iMGPjzQpkxTu_442GB4k",
    "authDomain": "vdiagn-6dded.firebaseapp.com",
    "projectId": "vdiagn-6dded",
    "storageBucket": "vdiagn-6dded.firebasestorage.app",
    "messagingSenderId": "454037057201",
    "appId": "1:454037057201:web:1bf8ba7dfd857a18143c90",
    "databaseURL": "https://vdiagn-6dded-default-rtdb.firebaseio.com/"
}

# Security Settings
SESSION_SECRET_KEY = "vehicle_diagnostics_secret_key_change_in_production_2025"
SESSION_PERMANENT = False
SESSION_TYPE = 'filesystem'

# App Configuration
DEBUG_MODE = True  # Set to False in production
FIREBASE_AVAILABLE = True  # Firebase is now properly configured

# Email Domain Restrictions (Optional)
ALLOWED_EMAIL_DOMAINS = []  # Empty list allows all domains

# Registration Settings
REQUIRE_EMAIL_VERIFICATION = False  # Set to True for production
AUTO_LOGIN_AFTER_REGISTRATION = True