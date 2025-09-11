
import os
from flask import Flask, render_template, request, jsonify
from nlp.diagnostic_processor import DiagnosticProcessor
import json

app = Flask(__name__)

# Initialize the diagnostic processor
try:
    obd_codes_path = os.path.join(os.path.dirname(__file__), 'data', 'obd_codes.json')
    diagnostic_processor = DiagnosticProcessor(obd_codes_path)
    print("Diagnostic processor initialized successfully")
except Exception as e:
    print(f"Error initializing diagnostic processor: {e}")
    diagnostic_processor = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/diagnose", methods=["POST"])
def diagnose():
    """Main endpoint for vehicle diagnostics"""
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()
        
        if not user_input:
            return jsonify({
                "error": "Please provide a description of your vehicle problem",
                "success": False
            })
        
        if diagnostic_processor is None:
            return jsonify({
                "error": "Diagnostic system is not available",
                "success": False
            })
        
        # Process the user input with NLP
        results = diagnostic_processor.process_user_input(user_input, top_n=5)
        
        if not results.get('matches'):
            return jsonify({
                "message": "No matching diagnostic codes found. Please provide more specific details about your vehicle problem.",
                "success": True,
                "matches": [],
                "analysis": results.get('analysis', {})
            })
        
        # Format the response
        formatted_matches = []
        for match in results['matches']:
            formatted_match = {
                "code": match['code_id'],
                "description": match['description'],
                "priority": match['priority'],
                "confidence": f"{match['confidence_score']:.1%}",
                "common_causes": match['common_causes'],
                "likely_cause": match['confirmation']
            }
            formatted_matches.append(formatted_match)
        
        response_data = {
            "success": True,
            "user_input": user_input,
            "matches": formatted_matches,
            "analysis": {
                "detected_symptoms": results['analysis'].get('detected_symptoms', []),
                "total_matches": len(formatted_matches)
            },
            "message": f"Found {len(formatted_matches)} potential diagnostic code(s) based on your description."
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in diagnose endpoint: {e}")
        return jsonify({
            "error": f"An error occurred during diagnosis: {str(e)}",
            "success": False
        })

@app.route("/get_code", methods=["POST"])
def get_code():
    """Get detailed information about a specific OBD code"""
    try:
        data = request.get_json()
        code_id = data.get("code_id", "").strip().upper()
        
        if not code_id:
            return jsonify({
                "error": "Please provide a valid OBD code ID",
                "success": False
            })
        
        if diagnostic_processor is None:
            return jsonify({
                "error": "Diagnostic system is not available",
                "success": False
            })
        
        # Get code details
        result = diagnostic_processor.get_code_details(code_id)
        
        if result['found']:
            code_data = result['code']
            return jsonify({
                "success": True,
                "code": {
                    "id": code_data['id'],
                    "description": code_data['description'],
                    "common_causes": code_data.get('common_causes', []),
                    "priority": code_data.get('priority', 'Unknown'),
                    "likely_cause": code_data.get('confirmation', 'Not specified')
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": result['message']
            })
            
    except Exception as e:
        print(f"Error in get_code endpoint: {e}")
        return jsonify({
            "error": f"An error occurred: {str(e)}",
            "success": False
        })

@app.route("/get_codes_by_priority", methods=["POST"])
def get_codes_by_priority():
    """Get all codes filtered by priority level"""
    try:
        data = request.get_json()
        priority = data.get("priority", "").strip()
        
        if priority.lower() not in ['high', 'medium', 'low']:
            return jsonify({
                "error": "Priority must be 'high', 'medium', or 'low'",
                "success": False
            })
        
        if diagnostic_processor is None:
            return jsonify({
                "error": "Diagnostic system is not available",
                "success": False
            })
        
        # Get codes by priority
        result = diagnostic_processor.get_all_codes_by_priority(priority)
        
        return jsonify({
            "success": True,
            "priority": result['priority'],
            "count": result['count'],
            "codes": result['codes']
        })
        
    except Exception as e:
        print(f"Error in get_codes_by_priority endpoint: {e}")
        return jsonify({
            "error": f"An error occurred: {str(e)}",
            "success": False
        })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    processor_status = "available" if diagnostic_processor is not None else "unavailable"
    return jsonify({
        "status": "healthy",
        "diagnostic_processor": processor_status,
        "total_codes": len(diagnostic_processor.obd_codes) if diagnostic_processor else 0
    })

# Legacy endpoint for backward compatibility
@app.route("/send_message", methods=["POST"])
def send_message():
    """Legacy endpoint - redirects to diagnose"""
    return diagnose()

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
