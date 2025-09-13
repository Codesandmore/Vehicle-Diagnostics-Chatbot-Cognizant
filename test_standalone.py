"""
Test the standalone vehicle diagnostics system
"""

import json

# Test the rule-based prediction function
def test_prediction():
    print("🧪 Testing Vehicle Diagnostics System")
    print("=" * 50)
    
    # Import the prediction function
    try:
        from standalone_demo import predict_anomaly_rule_based, NORMAL_RANGES
        
        # Test case 1: Normal values
        print("\n📊 Test 1: Normal Engine Parameters")
        normal_data = {
            "Vibration_Amplitude": 0.45,
            "RMS_Vibration": 0.12,
            "Vibration_Frequency": 300.0,
            "Surface_Temperature": 85.0,
            "Exhaust_Temperature": 450.0,
            "Acoustic_dB": 72.0,
            "Acoustic_Frequency": 1000.0,
            "Intake_Pressure": 1.2,
            "Exhaust_Pressure": 1.0,
            "Frequency_Band_Energy": 55.0,
            "Amplitude_Mean": 0.3
        }
        
        result1 = predict_anomaly_rule_based(normal_data)
        print(f"Status: {result1['status']}")
        print(f"Is Anomaly: {result1['is_anomaly']}")
        print(f"Risk Level: {result1['risk_level']}")
        
        # Test case 2: Anomalous values
        print("\n🚨 Test 2: Anomalous Engine Parameters")
        anomaly_data = {
            "Vibration_Amplitude": 1.5,  # Too high
            "RMS_Vibration": 0.4,        # Too high
            "Vibration_Frequency": 50.0,  # Too low
            "Surface_Temperature": 150.0,  # Too high
            "Exhaust_Temperature": 700.0,  # Too high
            "Acoustic_dB": 95.0,           # Too high
            "Acoustic_Frequency": 1000.0,
            "Intake_Pressure": 1.2,
            "Exhaust_Pressure": 1.0,
            "Frequency_Band_Energy": 55.0,
            "Amplitude_Mean": 0.3
        }
        
        result2 = predict_anomaly_rule_based(anomaly_data)
        print(f"Status: {result2['status']}")
        print(f"Is Anomaly: {result2['is_anomaly']}")
        print(f"Risk Level: {result2['risk_level']}")
        print(f"Issues Detected: {result2['total_anomalies']}")
        if result2['anomalies_detected']:
            for issue in result2['anomalies_detected']:
                print(f"  - {issue}")
        
        print("\n✅ System tests completed successfully!")
        print(f"📋 Normal ranges defined for {len(NORMAL_RANGES)} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_prediction()
    print("\n🚀 To start the web application, run:")
    print("   python standalone_demo.py")
    print("   Then open: http://localhost:5001")
