# 🚗 Standalone Vehicle Engine Diagnostics System

A completely independent web application for vehicle engine anomaly detection using machine learning.

## Features

- **Real-time Engine Analysis**: Input sensor readings and get instant anomaly detection
- **Beautiful Web Interface**: Modern, responsive design with intuitive form controls
- **Risk Assessment**: Categorizes detected anomalies as LOW, MEDIUM, or HIGH risk
- **Actionable Recommendations**: Provides specific next steps based on risk level
- **Standalone Operation**: Completely independent from existing chatbot system

## Quick Start

### 1. Install Dependencies
```bash
pip install -r standalone_requirements.txt
```

### 2. Run the Application
```bash
python standalone_app.py
```

### 3. Access the Application
Open your browser and go to: `http://localhost:5001`

## How to Use

1. **Enter Sensor Data**: Fill in all 11 engine sensor parameters
2. **Sample Data**: Click "Load Sample Data" to test with example values
3. **Analyze**: Click "Analyze Engine" to get results
4. **Review Results**: Get anomaly status, risk level, and recommendations

## Input Parameters

### Vibration Measurements
- **Vibration Amplitude**: Engine vibration magnitude
- **RMS Vibration**: Root mean square vibration value
- **Vibration Frequency**: Frequency of engine vibrations (Hz)

### Temperature Measurements
- **Surface Temperature**: Engine surface temperature (°C)
- **Exhaust Temperature**: Exhaust gas temperature (°C)

### Acoustic Measurements
- **Acoustic Level**: Engine noise level (dB)
- **Acoustic Frequency**: Dominant frequency in engine sound (Hz)

### Pressure Measurements
- **Intake Pressure**: Air intake pressure (bar)
- **Exhaust Pressure**: Exhaust back pressure (bar)

### Additional Measurements
- **Frequency Band Energy**: Energy in specific frequency bands
- **Amplitude Mean**: Average amplitude across measurements

## API Endpoints

### POST /api/diagnose
Analyze engine data for anomalies.

**Request Body:**
```json
{
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
```

**Response:**
```json
{
    "status": "NORMAL",
    "is_anomaly": false,
    "anomaly_score": -0.0234,
    "confidence": 0.0234,
    "risk_level": "LOW"
}
```

### GET /api/health
Check system health and model status.

## Model Information

- **Algorithm**: Isolation Forest (Unsupervised Anomaly Detection)
- **Training Data**: 10,000 engine sensor readings
- **Features**: 11 sensor parameters
- **Contamination Rate**: 3% (expected anomaly rate)
- **Model Persistence**: Automatically saves/loads trained models

## Files Structure

```
standalone_app.py              # Main Flask application
standalone_requirements.txt    # Python dependencies
standalone_templates/
    diagnostic_form.html       # Web interface
standalone_static/
    diagnostic_style.css       # Styling
    diagnostic_script.js       # Frontend logic
```

## Customization

### Adjusting Sensitivity
Modify the `CONTAMINATION` parameter in `standalone_app.py`:
```python
CONTAMINATION = 0.03  # 3% expected anomalies (less sensitive)
CONTAMINATION = 0.05  # 5% expected anomalies (more sensitive)
```

### Adding New Features
Add new sensor parameters to the `FEATURES` list and update the web form accordingly.

## Troubleshooting

1. **Port Already in Use**: Change port in `standalone_app.py`:
   ```python
   app.run(debug=True, host='0.0.0.0', port=5002)  # Use different port
   ```

2. **Model Training Issues**: Ensure the data file exists:
   ```
   data/engine_fault_detection_dataset.csv
   ```

3. **Missing Dependencies**: Install all requirements:
   ```bash
   pip install -r standalone_requirements.txt
   ```

## Development

To modify the system:
1. **Backend**: Edit `standalone_app.py`
2. **Frontend**: Edit files in `standalone_templates/` and `standalone_static/`
3. **Styling**: Modify `diagnostic_style.css`
4. **JavaScript**: Update `diagnostic_script.js`

---

🔧 **Independent System**: This application runs completely separately from your existing chatbot and can be deployed independently.
