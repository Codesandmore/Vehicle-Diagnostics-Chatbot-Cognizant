// Vehicle Engine Diagnostics JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('diagnosticForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const sampleBtn = document.getElementById('sampleBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const results = document.getElementById('results');
    const resultContent = document.getElementById('resultContent');

    // Sample data for testing
    const sampleData = {
        vibration_amplitude: 0.45,
        rms_vibration: 0.12,
        vibration_frequency: 300.0,
        surface_temperature: 85.0,
        exhaust_temperature: 450.0,
        acoustic_db: 72.0,
        acoustic_frequency: 1000.0,
        intake_pressure: 1.2,
        exhaust_pressure: 1.0,
        frequency_band_energy: 55.0,
        amplitude_mean: 0.3
    };

    // Load sample data
    sampleBtn.addEventListener('click', function() {
        Object.keys(sampleData).forEach(key => {
            const input = document.getElementById(key);
            if (input) {
                input.value = sampleData[key];
            }
        });
        
        // Add visual feedback
        sampleBtn.style.background = '#218838';
        sampleBtn.textContent = '✅ Sample Loaded';
        setTimeout(() => {
            sampleBtn.style.background = '#28a745';
            sampleBtn.textContent = '📝 Load Sample Data';
        }, 1500);
    });

    // Clear all fields
    clearBtn.addEventListener('click', function() {
        form.reset();
        results.classList.add('hidden');
    });

    // Form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading
        showLoading();
        
        // Collect form data
        const formData = new FormData(form);
        const data = {};
        
        // Convert form field names to API format
        const fieldMapping = {
            'vibration_amplitude': 'Vibration_Amplitude',
            'rms_vibration': 'RMS_Vibration',
            'vibration_frequency': 'Vibration_Frequency',
            'surface_temperature': 'Surface_Temperature',
            'exhaust_temperature': 'Exhaust_Temperature',
            'acoustic_db': 'Acoustic_dB',
            'acoustic_frequency': 'Acoustic_Frequency',
            'intake_pressure': 'Intake_Pressure',
            'exhaust_pressure': 'Exhaust_Pressure',
            'frequency_band_energy': 'Frequency_Band_Energy',
            'amplitude_mean': 'Amplitude_Mean'
        };

        // Get values from form inputs
        Object.keys(fieldMapping).forEach(fieldId => {
            const input = document.getElementById(fieldId);
            if (input && input.value) {
                data[fieldMapping[fieldId]] = parseFloat(input.value);
            }
        });

        try {
            const response = await fetch('/api/diagnose', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            
            if (response.ok) {
                showResults(result);
            } else {
                showError(result.error || 'An error occurred during analysis');
            }
            
        } catch (error) {
            showError('Network error: ' + error.message);
        } finally {
            hideLoading();
        }
    });

    function showLoading() {
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = '🔄 Analyzing...';
        loadingIndicator.classList.remove('hidden');
        results.classList.add('hidden');
    }

    function hideLoading() {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = '🔍 Analyze Engine';
        loadingIndicator.classList.add('hidden');
    }

    function showResults(result) {
        const isAnomaly = result.is_anomaly;
        const status = result.status;
        const score = result.anomaly_score;
        const confidence = result.confidence;
        const riskLevel = result.risk_level;
        const anomaliesDetected = result.anomalies_detected || [];
        const totalAnomalies = result.total_anomalies || 0;

        const cardClass = isAnomaly ? 'result-anomaly' : 'result-normal';
        const statusIcon = isAnomaly ? '⚠️' : '✅';
        
        let anomalyDetails = '';
        if (anomaliesDetected.length > 0) {
            anomalyDetails = `
                <div style="margin-top: 15px; text-align: left;">
                    <h4>Issues Detected:</h4>
                    <ul style="margin: 10px 0 0 20px;">
                        ${anomaliesDetected.map(anomaly => `<li style="margin: 5px 0;">${anomaly}</li>`).join('')}
                    </ul>
                    ${totalAnomalies > anomaliesDetected.length ? `<p><em>...and ${totalAnomalies - anomaliesDetected.length} more issues</em></p>` : ''}
                </div>
            `;
        }
        
        resultContent.innerHTML = `
            <div class="result-card ${cardClass}">
                <div class="result-title">
                    ${statusIcon} ${status}
                </div>
                <p>
                    ${isAnomaly ? 
                        'Potential engine issues detected. Review the details below.' : 
                        'All engine parameters are within normal operating range.'
                    }
                </p>
                
                <div class="result-details">
                    <div class="detail-item">
                        <div class="detail-label">Anomaly Score</div>
                        <div class="detail-value">${score}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Confidence</div>
                        <div class="detail-value">${confidence}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Risk Level</div>
                        <div class="detail-value risk-${riskLevel.toLowerCase()}">${riskLevel}</div>
                    </div>
                    ${totalAnomalies > 0 ? `
                    <div class="detail-item">
                        <div class="detail-label">Issues Found</div>
                        <div class="detail-value">${totalAnomalies}</div>
                    </div>` : ''}
                </div>
                
                ${anomalyDetails}
                ${isAnomaly ? getRecommendations(riskLevel) : ''}
            </div>
        `;
        
        results.classList.remove('hidden');
        results.scrollIntoView({ behavior: 'smooth' });
    }

    function getRecommendations(riskLevel) {
        const recommendations = {
            'HIGH': [
                'Stop vehicle operation immediately',
                'Schedule urgent inspection with a qualified mechanic',
                'Check for unusual noises, vibrations, or smells',
                'Do not drive until issue is resolved'
            ],
            'MEDIUM': [
                'Schedule inspection within the next few days',
                'Monitor engine performance closely',
                'Check fluid levels and engine condition',
                'Avoid heavy loads or extreme conditions'
            ],
            'LOW': [
                'Schedule routine maintenance check',
                'Continue normal operation with monitoring',
                'Keep records of engine parameters',
                'Consider preventive maintenance'
            ]
        };

        const recs = recommendations[riskLevel] || recommendations['LOW'];
        
        return `
            <div style="margin-top: 20px; text-align: left;">
                <h4>Recommended Actions:</h4>
                <ul style="margin: 10px 0 0 20px;">
                    ${recs.map(rec => `<li style="margin: 5px 0;">${rec}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    function showError(message) {
        resultContent.innerHTML = `
            <div class="error">
                <strong>Error:</strong> ${message}
            </div>
        `;
        results.classList.remove('hidden');
    }

    // Input validation
    const inputs = form.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            // Remove any invalid characters
            this.value = this.value.replace(/[^0-9.-]/g, '');
        });
    });
});
