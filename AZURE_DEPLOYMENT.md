# Azure Deployment Guide

## 🚀 Optimized for Azure App Service

This branch (`azure-deployment`) has been optimized for Azure deployment with reduced package size.

### ✅ What's Included:
- ✅ **NLP Vehicle Diagnosis** - Full functionality
- ✅ **Gemini AI Integration** - Chat and analysis 
- ✅ **Firebase Authentication** - User management
- ✅ **YouTube Video Search** - Diagnostic videos
- ✅ **All Core Features** - Complete vehicle diagnostics

### ❌ What's Removed for Size Optimization:
- ❌ **Speech Recognition** (OpenAI Whisper + PyTorch ~2GB+)
- ❌ **Heavy ML Libraries** (pandas, matplotlib, seaborn)
- ❌ **Audio Processing** (pydub, torchaudio)

### 📊 Size Reduction:
- **Before**: ~2.5GB with PyTorch/Whisper
- **After**: ~200MB optimized deployment
- **Reduction**: ~90% smaller

## 🔧 Deployment Steps:

### 1. **Local Testing**:
```bash
pip install -r requirements.txt
python app_combined.py
```

### 2. **Azure App Service Deployment**:
```bash
# Push this branch to your fork
git add .
git commit -m "Azure deployment optimization"
git push origin azure-deployment

# Deploy via Azure Portal or CLI
az webapp up --name your-app-name --resource-group your-rg
```

### 3. **Environment Variables** (Set in Azure Portal):
```
GOOGLE_API_KEY=your_gemini_api_key
SECRET_KEY=your_secret_key
FIREBASE_API_KEY=your_firebase_key
FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
FIREBASE_PROJECT_ID=your_project_id
YOUTUBE_API_KEY=your_youtube_key (optional)
```

## 🎯 Features Status:

| Feature | Status | Notes |
|---------|---------|--------|
| Vehicle Diagnosis | ✅ Full | NLP + AI analysis |
| Chat Interface | ✅ Full | Gemini AI powered |
| Image Analysis | ✅ Full | Upload & analyze vehicle images |
| User Authentication | ✅ Full | Firebase integration |
| YouTube Videos | ✅ Full | Diagnostic video search |
| Speech Recognition | ❌ Disabled | Reduced deployment size |
| Audio Upload | ❌ Disabled | Reduced deployment size |

## 🔄 Switching Back to Full Features:

If you need speech features locally, switch back to main:
```bash
git checkout main
pip install -r requirements.txt  # Installs full dependencies
```

## 📝 Notes:
- The app automatically detects missing speech dependencies
- Users will see a helpful message if they try to use speech features
- All core diagnostic functionality remains intact
- Perfect for production deployment on Azure App Service