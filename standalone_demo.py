"""
Demo Vehicle Diagnostics Backend (No ML Dependencies Required)
Standalone system with rule-based anomaly detection for demonstration
"""

try:
    from flask import Flask, request, jsonify, render_template
except ImportError:
    print("❌ Flask is required. Please install: pip install flask")
    print("💡 Run: pip install flask")
    exit(1)

import json
import math

app = Flask(__name__, 
           template_folder='standalone_templates',
           static_folder='standalone_static')

# Configuration
FEATURES = [
    "Vibration_Amplitude",
    "RMS_Vibration", 
    "Vibration_Frequency",
    "Surface_Temperature",
    "Exhaust_Temperature",
    "Acoustic_dB",
    "Acoustic_Frequency",
    "Intake_Pressure",
    "Exhaust_Pressure",
    "Frequency_Band_Energy",
    "Amplitude_Mean"
]

# Normal operating ranges for rule-based detection
NORMAL_RANGES = {
    "Vibration_Amplitude": (0.1, 0.8),
    "RMS_Vibration": (0.05, 0.25),
    "Vibration_Frequency": (100, 500),
    "Surface_Temperature": (60, 120),
    "Exhaust_Temperature": (200, 600),
    "Acoustic_dB": (50, 85),
    "Acoustic_Frequency": (500, 2000),
    "Intake_Pressure": (0.8, 2.0),
    "Exhaust_Pressure": (0.5, 1.5),
    "Frequency_Band_Energy": (20, 80),
    "Amplitude_Mean": (0.1, 0.6)
}

def predict_anomaly_rule_based(input_data):
    """Rule-based anomaly detection for demo purposes"""
    try:
        anomalies = []
        scores = []
        
        for feature in FEATURES:
            if feature not in input_data:
                return {"error": f"Missing feature: {feature}"}
            
            value = float(input_data[feature])
            min_val, max_val = NORMAL_RANGES[feature]
            
            # Calculate how far outside normal range
            if value < min_val:
                deviation = (min_val - value) / min_val
                anomalies.append(f"{feature} too low ({value:.2f} < {min_val})")
                scores.append(deviation)
            elif value > max_val:
                deviation = (value - max_val) / max_val
                anomalies.append(f"{feature} too high ({value:.2f} > {max_val})")
                scores.append(deviation)
            else:
                # Within normal range
                scores.append(0)
        
        # Calculate overall anomaly score
        max_deviation = max(scores) if scores else 0
        avg_deviation = sum(scores) / len(scores) if scores else 0
        
        # Determine if anomalous
        is_anomaly = len(anomalies) > 0 or max_deviation > 0.2
        anomaly_score = max_deviation + (avg_deviation * 0.3)
        
        # Determine status and risk
        if is_anomaly:
            if max_deviation > 0.5:
                risk_level = "HIGH"
                status = "CRITICAL ANOMALY DETECTED"
            elif max_deviation > 0.3:
                risk_level = "MEDIUM"
                status = "ANOMALY DETECTED"
            else:
                risk_level = "LOW"
                status = "MINOR ANOMALY DETECTED"
        else:
            risk_level = "LOW"
            status = "NORMAL"
        
        return {
            "status": status,
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "confidence": round(max_deviation, 4),
            "risk_level": risk_level,
            "anomalies_detected": anomalies[:3],  # Show top 3 issues
            "total_anomalies": len(anomalies)
        }
        
    except Exception as e:
        return {"error": str(e)}

def get_risk_level(score, is_anomaly):
    """Determine risk level based on anomaly score"""
    if not is_anomaly:
        return "LOW"
    elif score > 0.5:
        return "HIGH"
    elif score > 0.3:
        return "MEDIUM"
    else:
        return "LOW"

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('diagnostic_form.html')

@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    """API endpoint for vehicle diagnostics"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Predict anomaly using rule-based system
        result = predict_anomaly_rule_based(data)
        
        if "error" in result:
            return jsonify(result), 400
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "detection_method": "rule-based",
        "version": "demo-1.0"
    })

@app.route('/api/ranges', methods=['GET'])
def get_normal_ranges():
    """Get normal operating ranges for reference"""
    return jsonify({
        "normal_ranges": NORMAL_RANGES,
        "features": FEATURES
    })

if __name__ == '__main__':
    print("\n🚗 Vehicle Diagnostics System Started (Demo Mode)")
    print("📍 Access the application at: http://localhost:5001")
    print("🔧 API endpoint: http://localhost:5001/api/diagnose")
    print("📊 Normal ranges: http://localhost:5001/api/ranges")
    print("\n💡 This demo uses rule-based detection instead of ML")
    print("   Install scikit-learn and pandas for full ML version")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5001)
