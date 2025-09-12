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

nltk.download('punkt_tab')

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
            lowercase=True,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        
        # Fit vectorizer on all diagnostic texts
        all_texts = [text for text in self.searchable_texts.values()]
        if all_texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        else:
            self.tfidf_matrix = None
        
        # Vehicle symptom keywords mapping
        self.symptom_keywords = {
            'misfire': ['misfire', 'engine miss', 'rough idle', 'engine shake', 'rough running', 'stalling'],
            'oxygen_sensor': ['o2 sensor', 'oxygen sensor', 'lambda sensor', 'exhaust sensor', 'air fuel ratio'],
            'evap': ['evap', 'evaporative', 'emission', 'vapor', 'gas cap', 'fuel vapor', 'charcoal canister'],
            'temperature': ['temperature', 'thermostat', 'coolant', 'overheating', 'running cold', 'cooling system'],
            'knock': ['knock', 'pinging', 'engine knock', 'detonation', 'pre-ignition', 'rattling'],
            'fuel': ['fuel', 'gas', 'gasoline', 'fuel pump', 'injector', 'lean', 'rich', 'fuel system'],
            'ignition': ['ignition', 'spark plug', 'coil', 'distributor', 'firing', 'no start'],
            'exhaust': ['exhaust', 'catalytic converter', 'cat', 'emissions', 'muffler'],
            'vacuum': ['vacuum', 'vacuum leak', 'air leak', 'intake', 'manifold'],
            'electrical': ['electrical', 'wiring', 'connection', 'circuit', 'short', 'battery', 'alternator'],
            'transmission': ['transmission', 'shifting', 'gear', 'clutch', 'torque converter'],
            'brakes': ['brake', 'braking', 'abs', 'brake light', 'brake pedal'],
            'engine_performance': ['power loss', 'acceleration', 'performance', 'sluggish', 'lack of power']
        }
        
        # Non-diagnostic phrases that should be filtered out
        self.non_diagnostic_phrases = {
            'greetings': ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'],
            'questions': ['what is', 'how do', 'can you', 'please help', 'i need help'],
            'general': ['thanks', 'thank you', 'ok', 'okay', 'yes', 'no', 'maybe']
        }
        
        # Minimum input requirements
        self.min_word_count = 3
        self.min_meaningful_words = 2
    
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
    
    def _is_valid_diagnostic_input(self, user_input):
        """Validate if the input is suitable for diagnostic processing"""
        if not user_input or not user_input.strip():
            return False, "Empty input provided"
        
        # Clean and tokenize the input
        cleaned_input = user_input.lower().strip()
        tokens = word_tokenize(cleaned_input)
        
        # Check for obvious non-diagnostic phrases (more lenient)
        for category, phrases in self.non_diagnostic_phrases.items():
            for phrase in phrases:
                if phrase == cleaned_input.strip():  # Exact match only
                    return False, f"Input appears to be a {category} rather than a vehicle diagnostic issue"
        
        # Very relaxed minimum requirements
        if len(tokens) < 2:  # Reduced from 3
            return False, "Input too short. Please provide more details about your vehicle issue"
        
        # Enhanced automotive keyword detection (more flexible)
        automotive_keywords = set()
        for symptom_list in self.symptom_keywords.values():
            automotive_keywords.update(symptom_list)
        
        # Extended automotive vocabulary with related terms
        automotive_keywords.update([
            'car', 'vehicle', 'engine', 'motor', 'transmission', 'brake', 'wheel', 
            'tire', 'battery', 'starter', 'alternator', 'radiator', 'exhaust',
            'oil', 'coolant', 'fluid', 'light', 'dashboard', 'check engine',
            'rpm', 'idle', 'acceleration', 'speed', 'gear', 'clutch',
            'driving', 'drive', 'start', 'starting', 'stop', 'stopping',
            'sound', 'noise', 'vibration', 'smoke', 'leak', 'warning',
            'problem', 'issue', 'trouble', 'error', 'fault', 'malfunction',
            'rough', 'hard', 'difficult', 'won\'t', 'wont', 'cant', 'can\'t',
            'fuel', 'gas', 'diesel', 'mileage', 'consumption', 'performance'
        ])
        
        # Check for automotive context (partial matches allowed)
        has_automotive_context = any(
            keyword in cleaned_input or any(word in keyword for word in tokens if len(word) > 3)
            for keyword in automotive_keywords
        )
        
        # If no direct automotive keywords, check for problem-indicating words
        problem_indicators = ['problem', 'issue', 'trouble', 'error', 'fault', 'wrong', 'broken', 'not working', 'wont work', 'difficult', 'hard']
        has_problem_context = any(indicator in cleaned_input for indicator in problem_indicators)
        
        # Accept input if it has either automotive context OR problem context (very lenient)
        if not (has_automotive_context or has_problem_context):
            return False, "Please describe a vehicle problem or automotive issue"
        
        return True, "Valid diagnostic input"
    
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
        if self.tfidf_matrix is None:
            return []
            
        # Preprocess user input
        processed_input = self._preprocess_text(user_input)
        if not processed_input.strip():
            return []
            
        try:
            # Transform user input using fitted vectorizer
            user_vector = self.vectorizer.transform([processed_input])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
            
            # Create results list with code IDs and similarities
            results = []
            code_ids = list(self.searchable_texts.keys())
            
            for i, similarity in enumerate(similarities):
                if similarity > 0.05:  # Lowered threshold from 0.1 to 0.05
                    results.append({
                        'code_id': code_ids[i],
                        'similarity': float(similarity),  # Ensure it's a float
                        'method': 'tfidf'
                    })
            
            return sorted(results, key=lambda x: x['similarity'], reverse=True)
        except Exception as e:
            print(f"Error in TF-IDF calculation: {e}")
            return []
    
    def _calculate_fuzzy_similarity(self, user_input):
        """Calculate fuzzy string similarity for exact matches"""
        results = []
        user_input_lower = user_input.lower().strip()
        
        # Skip if input is too short for meaningful fuzzy matching
        if len(user_input_lower) < 3:  # Reduced from 5 to 3
            return []
        
        for code_id, searchable_text in self.searchable_texts.items():
            # Calculate different fuzzy matching scores
            ratio_score = fuzz.ratio(user_input_lower, searchable_text) / 100.0
            partial_score = fuzz.partial_ratio(user_input_lower, searchable_text) / 100.0
            token_sort_score = fuzz.token_sort_ratio(user_input_lower, searchable_text) / 100.0
            token_set_score = fuzz.token_set_ratio(user_input_lower, searchable_text) / 100.0
            
            # Use weighted average, giving more weight to partial matches for flexibility
            weighted_score = (
                ratio_score * 0.2 + 
                partial_score * 0.5 +  # Increased weight for partial matches
                token_sort_score * 0.2 + 
                token_set_score * 0.1
            )
            
            # More lenient threshold for fuzzy matching
            if weighted_score > 0.25:  # Reduced threshold from 0.4 to 0.25
                results.append({
                    'code_id': code_id,
                    'similarity': weighted_score,
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
        """Combine and rank results from different methods with proper normalization"""
        # Create a dictionary to combine scores
        combined_scores = {}
        
        # Add TF-IDF scores (weight: 0.5, max contribution: 0.5)
        for result in tfidf_results[:10]:  # Top 10 TF-IDF results
            code_id = result['code_id']
            normalized_score = min(result['similarity'], 1.0)  # Cap at 1.0
            combined_scores[code_id] = combined_scores.get(code_id, 0) + (normalized_score * 0.5)
        
        # Add fuzzy scores (weight: 0.3, max contribution: 0.3)
        for result in fuzzy_results[:10]:  # Top 10 fuzzy results
            code_id = result['code_id']
            normalized_score = min(result['similarity'], 1.0)  # Cap at 1.0
            combined_scores[code_id] = combined_scores.get(code_id, 0) + (normalized_score * 0.3)
        
        # Add symptom matching bonus (weight: 0.2, max contribution: 0.2)
        for code_id in combined_scores:
            code_data = self._find_code_by_id(code_id)
            if code_data:
                searchable_text = self.searchable_texts[code_id]
                symptom_matches = 0
                total_symptoms = max(len(symptoms), 1)  # Avoid division by zero
                
                for symptom in symptoms:
                    symptom_keywords = self.symptom_keywords.get(symptom, [])
                    for keyword in symptom_keywords:
                        if keyword in searchable_text:
                            symptom_matches += 1
                            break  # Only count each symptom once
                
                # Calculate symptom bonus (0.0 to 0.2)
                symptom_bonus = (symptom_matches / total_symptoms) * 0.2
                combined_scores[code_id] += symptom_bonus
        
        # Ensure no score exceeds 1.0 (100%)
        for code_id in combined_scores:
            combined_scores[code_id] = min(combined_scores[code_id], 1.0)
        
        # Sort by combined score and return only scores above minimum threshold
        min_threshold = 0.08  # Reduced from 0.15 to 0.08 - minimum 8% confidence required
        ranked_results = [
            (code_id, score) for code_id, score in combined_scores.items() 
            if score >= min_threshold
        ]
        
        return sorted(ranked_results, key=lambda x: x[1], reverse=True)
    
    def process_user_input(self, user_input, top_n=5):
        """
        Main method to process user input and return matching OBD codes
        
        Args:
            user_input (str): User's description of the vehicle problem
            top_n (int): Number of top results to return
            
        Returns:
            dict: Contains matched codes, confidence scores, and analysis
        """
        # Validate input
        is_valid, validation_message = self._is_valid_diagnostic_input(user_input)
        if not is_valid:
            return {
                'error': validation_message,
                'matches': [],
                'analysis': {
                    'validation_failed': True,
                    'reason': validation_message
                }
            }
        
        # Extract symptoms
        symptoms = self._extract_symptoms(user_input)
        
        # Calculate similarities using different methods
        tfidf_results = self._calculate_tfidf_similarity(user_input)
        fuzzy_results = self._calculate_fuzzy_similarity(user_input)
        
        # If no meaningful results from either method, return early
        if not tfidf_results and not fuzzy_results:
            return {
                'error': 'No matching diagnostic codes found. Your symptoms might be too general or not in our database.',
                'matches': [],
                'analysis': {
                    'detected_symptoms': symptoms,
                    'total_codes_analyzed': len(self.obd_codes),
                    'tfidf_matches': 0,
                    'fuzzy_matches': 0,
                    'user_input_processed': self._preprocess_text(user_input)
                }
            }
        
        # Rank and combine results
        ranked_results = self._rank_results(tfidf_results, fuzzy_results, symptoms)
        
        # If no results meet minimum threshold, return early
        if not ranked_results:
            return {
                'error': 'No diagnostic codes found with sufficient confidence. Please provide more specific details.',
                'matches': [],
                'analysis': {
                    'detected_symptoms': symptoms,
                    'total_codes_analyzed': len(self.obd_codes),
                    'tfidf_matches': len(tfidf_results),
                    'fuzzy_matches': len(fuzzy_results),
                    'user_input_processed': self._preprocess_text(user_input),
                    'threshold_not_met': True
                }
            }
        
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
                    'confidence_score': round(combined_score, 3)  # Already capped at 1.0
                })
        
        return {
            'matches': matches,
            'analysis': {
                'detected_symptoms': symptoms,
                'total_codes_analyzed': len(self.obd_codes),
                'tfidf_matches': len(tfidf_results),
                'fuzzy_matches': len(fuzzy_results),
                'user_input_processed': self._preprocess_text(user_input),
                'validation_passed': True
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
