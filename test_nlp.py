#!/usr/bin/env python3
"""
Test script to verify NLP system is properly using the full OBD codes database
"""

import os
from nlp.enhanced_diagnostic_processor import EnhancedDiagnosticProcessor

def test_nlp_system():
    # Initialize enhanced processor
    obd_codes_path = os.path.join(os.path.dirname(__file__), 'data', 'obd_codes.json')
    processor = EnhancedDiagnosticProcessor(obd_codes_path)
    
    print(f"🔍 Loaded {len(processor.obd_codes)} OBD codes from JSON file")
    print(f"🔍 Created {len(processor.searchable_texts)} searchable texts")
    print(f"🔍 TF-IDF Matrix shape: {processor.tfidf_matrix.shape if processor.tfidf_matrix is not None else 'None'}")
    
    # Show sample of loaded data
    print(f"\n📋 Sample OBD codes loaded:")
    for i, code in enumerate(processor.obd_codes[:5]):  # Show first 5
        print(f"   {i+1}. {code['id']}: {code['description']}")
    
    # Test with oxygen sensor queries
    test_queries = [
        "oxygen sensor malfunction",
        "o2 sensor circuit malfunction",
        "faulty shift solenoid",
        "engine misfire",
        "fuel system rich"
    ]
    
    print(f"\n🧪 Testing queries:")
    for query in test_queries:
        print(f"\n--- Testing: '{query}' ---")
        
        # Test TF-IDF matching
        tfidf_results = processor._calculate_tfidf_similarity(query)
        print(f"TF-IDF found {len(tfidf_results)} matches")
        
        # Test fuzzy matching  
        fuzzy_results = processor._calculate_fuzzy_similarity(query)
        print(f"Fuzzy found {len(fuzzy_results)} matches")
        
        # Test full processing
        full_results = processor.process_user_input(query, top_n=3)
        print(f"Full processing result:")
        print(f"  - Success: {full_results.get('success', 'N/A')}")
        print(f"  - Route to LLM only: {full_results.get('route_to_llm_only', 'N/A')}")
        print(f"  - Max confidence: {full_results.get('max_confidence', 0):.1f}%")
        print(f"  - Matches found: {len(full_results.get('matches', []))}")
        
        if full_results.get('matches'):
            for match in full_results['matches'][:2]:  # Show top 2
                print(f"    • {match['code_id']}: {match['description'][:50]}...")

if __name__ == "__main__":
    test_nlp_system()