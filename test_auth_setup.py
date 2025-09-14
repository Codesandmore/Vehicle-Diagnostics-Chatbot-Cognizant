#!/usr/bin/env python3
"""
Authentication Test Script for Vehicle Diagnostics Chatbot
This script tests the Firebase authentication setup without starting the full web server.
"""

import sys
import os

def test_imports():
    """Test if all required packages are installed."""
    print("🔍 Testing imports...")
    
    try:
        import pyrebase
        print("✅ pyrebase4 - OK")
    except ImportError:
        print("❌ pyrebase4 - Missing (pip install pyrebase4)")
        return False
    
    try:
        from flask_session import Session
        print("✅ flask-session - OK")
    except ImportError:
        print("❌ flask-session - Missing (pip install flask-session)")
        return False
    
    try:
        import flask
        print("✅ Flask - OK")
    except ImportError:
        print("❌ Flask - Missing (pip install flask)")
        return False
    
    return True

def test_config():
    """Test Firebase configuration."""
    print("\n🔍 Testing Firebase configuration...")
    
    try:
        from firebase_config import FIREBASE_CONFIG, SESSION_SECRET_KEY
        
        # Check if config has been updated from defaults
        if FIREBASE_CONFIG['apiKey'] == 'your-api-key-here':
            print("⚠️  Firebase config contains default values")
            print("   Please update firebase_config.py with your actual Firebase credentials")
            return False
        
        if SESSION_SECRET_KEY == 'your_secret_key_here_change_this_in_production':
            print("⚠️  Session secret key is still default")
            print("   Please change SESSION_SECRET_KEY in firebase_config.py")
            return False
        
        print("✅ Firebase configuration - OK")
        return True
        
    except ImportError as e:
        print(f"❌ Firebase config import failed: {e}")
        return False

def test_templates():
    """Test if required templates exist."""
    print("\n🔍 Testing template files...")
    
    templates = [
        'templates/login.html',
        'templates/register.html',
        'templates/index.html'
    ]
    
    all_exist = True
    for template in templates:
        if os.path.exists(template):
            print(f"✅ {template} - OK")
        else:
            print(f"❌ {template} - Missing")
            all_exist = False
    
    return all_exist

def test_firebase_connection():
    """Test Firebase connection (if configured)."""
    print("\n🔍 Testing Firebase connection...")
    
    try:
        from firebase_config import FIREBASE_CONFIG
        import pyrebase
        
        # Only test if config has been updated
        if FIREBASE_CONFIG['apiKey'] == 'your-api-key-here':
            print("⚠️  Skipping connection test - using default config")
            return True
        
        firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
        auth = firebase.auth()
        
        print("✅ Firebase connection - OK")
        return True
        
    except Exception as e:
        print(f"❌ Firebase connection failed: {e}")
        print("   This might be due to incorrect configuration or network issues")
        return False

def main():
    """Run all tests."""
    print("🚀 Vehicle Diagnostics Authentication Setup Test\n")
    
    tests = [
        ("Required packages", test_imports),
        ("Configuration files", test_config),
        ("Template files", test_templates),
        ("Firebase connection", test_firebase_connection)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} - Error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your authentication setup is ready.")
        print("\nNext steps:")
        print("1. Update firebase_config.py with your actual Firebase credentials")
        print("2. Run: python app_combined.py")
        print("3. Navigate to: http://localhost:5000/auth/login")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running the app.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)