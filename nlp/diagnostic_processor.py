"""
Vehicle Diagnostic NLP Processor
This module processes user input using NLP techniques to match with OBD codes
"""

import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process
import numpy as np
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class DiagnosticProcessor:
    def __init__(self, obd_codes_path):
        """Initialize the diagnostic processor with OBD codes data"""
        self.obd_codes_path = obd_codes_path
        self.obd_codes = self._load_obd_codes()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Create searchable text for each OBD code
        self.searchable_texts = self._create_searchable_texts()
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),
            max_features=5000,
            lowercase=True
        )
        
        # Fit vectorizer on all diagnostic texts
        all_texts = [text for text in self.searchable_texts.values()]
        self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Vehicle symptom keywords mapping
        self.symptom_keywords = {
            'misfire': ['misfire', 'engine miss', 'rough idle', 'engine shake', 'rough running'],
            'oxygen_sensor': ['o2 sensor', 'oxygen sensor', 'lambda sensor', 'exhaust sensor'],
            'evap': ['evap', 'evaporative', 'emission', 'vapor', 'gas cap', 'fuel vapor'],
            'temperature': ['temperature', 'thermostat', 'coolant', 'overheating', 'running cold'],
            'knock': ['knock', 'pinging', 'engine knock', 'detonation', 'pre-ignition'],
            'fuel': ['fuel', 'gas', 'gasoline', 'fuel pump', 'injector', 'lean', 'rich'],
            'ignition': ['ignition', 'spark plug', 'coil', 'distributor', 'firing'],
            'exhaust': ['exhaust', 'catalytic converter', 'cat', 'emissions'],
            'vacuum': ['vacuum', 'vacuum leak', 'air leak', 'intake'],
            'electrical': ['electrical', 'wiring', 'connection', 'circuit', 'short']
        }
    
    def _load_obd_codes(self):
        """Load OBD codes from JSON file"""
        try:
            with open(self.obd_codes_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: OBD codes file not found at {self.obd_codes_path}")
            return []
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in OBD codes file")
            return []
    
    def _create_searchable_texts(self):
        """Create searchable text for each OBD code combining all relevant information"""
        searchable_texts = {}
        
        for code in self.obd_codes:
            # Combine description, common causes, and confirmation into searchable text
            text_parts = [code.get('description', '')]
            
            # Add common causes
            if 'common_causes' in code and isinstance(code['common_causes'], list):
                text_parts.extend(code['common_causes'])
            
            # Add confirmation text
            if 'confirmation' in code:
                text_parts.append(code['confirmation'])
            
            # Join all parts and clean
            full_text = ' '.join(text_parts).lower()
            searchable_texts[code['id']] = full_text
        
        return searchable_texts
    
    def _preprocess_text(self, text):
        """Preprocess text for NLP analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers (keep spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def _extract_symptoms(self, user_input):
        """Extract vehicle symptoms from user input"""
        user_input_lower = user_input.lower()
        detected_symptoms = []
        
        for symptom_category, keywords in self.symptom_keywords.items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    detected_symptoms.append(symptom_category)
                    break
        
        return detected_symptoms
    
    def _calculate_tfidf_similarity(self, user_input):
        """Calculate TF-IDF cosine similarity between user input and OBD codes"""
        # Preprocess user input
        processed_input = self._preprocess_text(user_input)
        
        # Transform user input using fitted vectorizer
        user_vector = self.vectorizer.transform([processed_input])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        
        # Create results list with code IDs and similarities
        results = []
        code_ids = list(self.searchable_texts.keys())
        
        for i, similarity in enumerate(similarities):
            if similarity > 0.1:  # Only include meaningful similarities
                results.append({
                    'code_id': code_ids[i],
                    'similarity': similarity,
                    'method': 'tfidf'
                })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    def _calculate_fuzzy_similarity(self, user_input):
        """Calculate fuzzy string similarity for exact matches"""
        results = []
        user_input_lower = user_input.lower()
        
        for code_id, searchable_text in self.searchable_texts.items():
            # Calculate different fuzzy matching scores
            ratio_score = fuzz.ratio(user_input_lower, searchable_text) / 100.0
            partial_score = fuzz.partial_ratio(user_input_lower, searchable_text) / 100.0
            token_sort_score = fuzz.token_sort_ratio(user_input_lower, searchable_text) / 100.0
            token_set_score = fuzz.token_set_ratio(user_input_lower, searchable_text) / 100.0
            
            # Use the highest score
            max_score = max(ratio_score, partial_score, token_sort_score, token_set_score)
            
            if max_score > 0.3:  # Only include meaningful matches
                results.append({
                    'code_id': code_id,
                    'similarity': max_score,
                    'method': 'fuzzy'
                })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    def _find_code_by_id(self, code_id):
        """Find OBD code details by ID"""
        for code in self.obd_codes:
            if code['id'] == code_id:
                return code
        return None
    
    def _rank_results(self, tfidf_results, fuzzy_results, symptoms):
        """Combine and rank results from different methods"""
        # Create a dictionary to combine scores
        combined_scores = {}
        
        # Add TF-IDF scores (weight: 0.6)
        for result in tfidf_results[:10]:  # Top 10 TF-IDF results
            code_id = result['code_id']
            combined_scores[code_id] = combined_scores.get(code_id, 0) + (result['similarity'] * 0.6)
        
        # Add fuzzy scores (weight: 0.4)
        for result in fuzzy_results[:10]:  # Top 10 fuzzy results
            code_id = result['code_id']
            combined_scores[code_id] = combined_scores.get(code_id, 0) + (result['similarity'] * 0.4)
        
        # Boost scores for symptom matches
        for code_id in combined_scores:
            code_data = self._find_code_by_id(code_id)
            if code_data:
                searchable_text = self.searchable_texts[code_id]
                for symptom in symptoms:
                    symptom_keywords = self.symptom_keywords.get(symptom, [])
                    for keyword in symptom_keywords:
                        if keyword in searchable_text:
                            combined_scores[code_id] += 0.2  # Symptom boost
                            break
        
        # Sort by combined score
        ranked_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return ranked_results
    
    def process_user_input(self, user_input, top_n=5):
        """
        Main method to process user input and return matching OBD codes
        
        Args:
            user_input (str): User's description of the vehicle problem
            top_n (int): Number of top results to return
            
        Returns:
            dict: Contains matched codes, confidence scores, and analysis
        """
        if not user_input or not user_input.strip():
            return {
                'error': 'Empty input provided',
                'matches': [],
                'analysis': {}
            }
        
        # Extract symptoms
        symptoms = self._extract_symptoms(user_input)
        
        # Calculate similarities using different methods
        tfidf_results = self._calculate_tfidf_similarity(user_input)
        fuzzy_results = self._calculate_fuzzy_similarity(user_input)
        
        # Rank and combine results
        ranked_results = self._rank_results(tfidf_results, fuzzy_results, symptoms)
        
        # Prepare final results
        matches = []
        for code_id, combined_score in ranked_results[:top_n]:
            code_data = self._find_code_by_id(code_id)
            if code_data:
                matches.append({
                    'code_id': code_id,
                    'description': code_data.get('description', ''),
                    'common_causes': code_data.get('common_causes', []),
                    'priority': code_data.get('priority', 'Unknown'),
                    'confirmation': code_data.get('confirmation', ''),
                    'confidence_score': round(combined_score, 3)
                })
        
        return {
            'matches': matches,
            'analysis': {
                'detected_symptoms': symptoms,
                'total_codes_analyzed': len(self.obd_codes),
                'tfidf_matches': len(tfidf_results),
                'fuzzy_matches': len(fuzzy_results),
                'user_input_processed': self._preprocess_text(user_input)
            }
        }
    
    def get_code_details(self, code_id):
        """Get detailed information about a specific OBD code"""
        code_data = self._find_code_by_id(code_id)
        if code_data:
            return {
                'found': True,
                'code': code_data
            }
        else:
            return {
                'found': False,
                'message': f'OBD code {code_id} not found in database'
            }
    
    def get_all_codes_by_priority(self, priority):
        """Get all codes filtered by priority level"""
        filtered_codes = [
            code for code in self.obd_codes 
            if code.get('priority', '').lower() == priority.lower()
        ]
        
        return {
            'priority': priority,
            'count': len(filtered_codes),
            'codes': filtered_codes
        }


def main():
    """Test the diagnostic processor"""
    # Initialize processor
    obd_codes_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'newdata.json')
    processor = DiagnosticProcessor(obd_codes_path)
    
    # Test with sample inputs
    test_inputs = [
        "My car has a rough idle and engine misfire",
        "Check engine light is on, oxygen sensor issue",
        "Gas cap is loose and I smell fuel vapors",
        "Engine is running cold, thermostat problem",
        "Hearing knocking sounds from engine",
    ]
    
    print("=== Vehicle Diagnostic NLP Processor Test ===\n")
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"Test {i}: '{test_input}'")
        results = processor.process_user_input(test_input, top_n=3)
        
        print(f"Detected symptoms: {results['analysis']['detected_symptoms']}")
        print("Top matches:")
        
        for j, match in enumerate(results['matches'], 1):
            print(f"  {j}. {match['code_id']}: {match['description']}")
            print(f"     Priority: {match['priority']}, Confidence: {match['confidence_score']}")
        
        print("-" * 50)


if __name__ == "__main__":
    main()
