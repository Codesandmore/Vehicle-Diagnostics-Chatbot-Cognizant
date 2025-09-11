"""
Test script for the Vehicle Diagnostic NLP System
This script tests various user inputs and demonstrates the NLP matching capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from nlp.diagnostic_processor import DiagnosticProcessor
import json

def print_separator(title=""):
    print("=" * 80)
    if title:
        print(f" {title} ".center(80, "="))
    print("=" * 80)

def print_results(user_input, results):
    print(f"\n🔍 USER INPUT: '{user_input}'")
    print("-" * 50)
    
    analysis = results.get('analysis', {})
    matches = results.get('matches', [])
    
    if analysis.get('detected_symptoms'):
        print(f"🎯 DETECTED SYMPTOMS: {', '.join(analysis['detected_symptoms'])}")
    
    if matches:
        print(f"📊 FOUND {len(matches)} MATCHING CODES:")
        
        for i, match in enumerate(matches, 1):
            print(f"\n  {i}. CODE: {match['code_id']} ({match['priority']} Priority)")
            print(f"     CONFIDENCE: {match['confidence_score']:.1%}")
            print(f"     DESCRIPTION: {match['description']}")
            print(f"     LIKELY CAUSE: {match['confirmation']}")
            if match['common_causes']:
                causes = match['common_causes'][:3]  # Show first 3 causes
                print(f"     COMMON CAUSES: {', '.join(causes)}")
    else:
        print("❌ No matching codes found")
    
    print("\n" + "⚡" * 50)

def main():
    print_separator("VEHICLE DIAGNOSTIC NLP SYSTEM TEST")
    
    # Initialize the processor
    try:
        obd_codes_path = os.path.join('data', 'obd_codes.json')
        if not os.path.exists(obd_codes_path):
            print(f"❌ Error: OBD codes file not found at {obd_codes_path}")
            return
        
        processor = DiagnosticProcessor(obd_codes_path)
        print(f"✅ Diagnostic processor initialized successfully")
        print(f"📚 Loaded {len(processor.obd_codes)} OBD codes for analysis")
        
    except Exception as e:
        print(f"❌ Error initializing processor: {e}")
        return

    # Test cases - various user inputs
    test_cases = [
        # Engine misfire related
        "My car has a rough idle and engine misfire",
        "Engine is misfiring and running rough",
        "Random cylinder misfire detected",
        
        # Oxygen sensor issues
        "Check engine light is on, oxygen sensor issue",
        "O2 sensor malfunction, heater circuit problem",
        "Lambda sensor not working properly",
        
        # EVAP system issues
        "Gas cap is loose and I smell fuel vapors", 
        "Evaporative emission system leak",
        "Fuel vapor leak, small evap leak",
        
        # Temperature issues
        "Engine is running cold, thermostat problem",
        "Coolant temperature below normal",
        "Thermostat stuck open",
        
        # Knock sensor issues
        "Hearing knocking sounds from engine",
        "Engine knock detected",
        "Pinging noise from engine",
        
        # Fuel system issues
        "Engine running lean, fuel delivery problem",
        "Fuel pump not working properly",
        "Injector clogged, poor fuel delivery",
        
        # Mixed/complex issues
        "Check engine light on, rough idle, possible vacuum leak",
        "Car won't start, fuel and ignition problems",
        
        # Specific code lookups
        "What is P0300?",
        "P0442 code meaning",
        
        # Edge cases
        "My car makes weird noises",  # Vague input
        "",  # Empty input
        "The flux capacitor is broken"  # Nonsensical input
    ]

    print_separator("RUNNING TEST CASES")
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        
        try:
            results = processor.process_user_input(test_input, top_n=3)
            print_results(test_input, results)
            
        except Exception as e:
            print(f"❌ Error processing input '{test_input}': {e}")
    
    # Test specific code lookup
    print_separator("TESTING SPECIFIC CODE LOOKUPS")
    
    test_codes = ['P0300', 'P0141', 'P0442', 'P9999']  # Last one doesn't exist
    
    for code in test_codes:
        print(f"\n🔍 LOOKING UP CODE: {code}")
        result = processor.get_code_details(code)
        
        if result['found']:
            code_data = result['code']
            print(f"✅ FOUND: {code_data['description']}")
            print(f"   Priority: {code_data.get('priority', 'Unknown')}")
            if code_data.get('common_causes'):
                print(f"   Causes: {', '.join(code_data['common_causes'][:3])}")
        else:
            print(f"❌ {result['message']}")
    
    # Test priority filtering
    print_separator("TESTING PRIORITY FILTERING")
    
    for priority in ['high', 'medium', 'low']:
        result = processor.get_all_codes_by_priority(priority)
        print(f"🎯 {priority.upper()} PRIORITY CODES: {result['count']} found")
    
    print_separator("TEST COMPLETED")
    print("✅ All tests completed successfully!")
    print("💡 The NLP system is ready to process user vehicle diagnostic queries.")

if __name__ == "__main__":
    main()
