"""
Enhanced Vehicle Diagnostic NLP Processor with Confidence Threshold System
This module processes user input using NLP techniques with enhanced confidence calculation
and intelligent routing to LLM based on confidence thresholds.
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


class EnhancedDiagnosticProcessor:
    def __init__(self, obd_codes_path):
        """Initialize the enhanced diagnostic processor with OBD codes data"""
        self.obd_codes_path = obd_codes_path
        self.obd_codes = self._load_obd_codes()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Create searchable text for each OBD code
        self.searchable_texts = self._create_searchable_texts()
        
        # Initialize TF-IDF vectorizer with enhanced parameters
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),
            max_features=5000,
            lowercase=True,
            min_df=2,
            max_df=0.8
        )
        
        # Fit vectorizer on all diagnostic texts
        all_texts = [text for text in self.searchable_texts.values()]
        if all_texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        else:
            self.tfidf_matrix = None
        
        # Enhanced vehicle symptom keywords mapping
        self.symptom_keywords = {
            'misfire': ['misfire', 'engine miss', 'rough idle', 'engine shake', 'rough running', 'stalling', 'cylinder miss'],
            'oxygen_sensor': ['o2 sensor', 'oxygen sensor', 'lambda sensor', 'exhaust sensor', 'air fuel ratio', 'lean', 'rich'],
            'evap': ['evap', 'evaporative', 'emission', 'vapor', 'gas cap', 'fuel vapor', 'charcoal canister'],
            'temperature': ['temperature', 'thermostat', 'coolant', 'overheating', 'running cold', 'cooling system'],
            'knock': ['knock', 'pinging', 'engine knock', 'detonation', 'pre-ignition', 'rattling'],
            'fuel': ['fuel', 'gas', 'gasoline', 'fuel pump', 'injector', 'lean', 'rich', 'fuel system', 'starvation'],
            'ignition': ['ignition', 'spark plug', 'coil', 'distributor', 'firing', 'no start', 'misfire'],
            'exhaust': ['exhaust', 'catalytic converter', 'cat', 'emissions', 'muffler'],
            'vacuum': ['vacuum', 'vacuum leak', 'air leak', 'intake', 'manifold'],
            'electrical': ['electrical', 'wiring', 'connection', 'circuit', 'short', 'battery', 'alternator'],
            'transmission': ['transmission', 'shifting', 'gear', 'clutch', 'torque converter'],
            'brakes': ['brake', 'braking', 'abs', 'brake light', 'brake pedal'],
            'engine_performance': ['power loss', 'acceleration', 'performance', 'sluggish', 'lack of power']
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
        """Create searchable text combining description and causes for each OBD code"""
        searchable_texts = {}
        for code in self.obd_codes:
            code_id = code['id']
            description = code.get('description', '')
            common_causes = ' '.join(code.get('common_causes', []))
            searchable_text = f"{description} {common_causes}".lower()
            searchable_texts[code_id] = searchable_text
        return searchable_texts
    
    def _preprocess_text(self, text):
        """Preprocess text for TF-IDF analysis"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def _calculate_enhanced_confidence(self, user_input, code_data, tfidf_score=0, fuzzy_score=0, symptoms_matched=[]):
        """
        Enhanced confidence calculation with multiple factors
        Returns a confidence score between 0 and 100
        """
        confidence_factors = {}
        
        # 1. Base semantic similarity (40% weight)
        semantic_score = max(tfidf_score, fuzzy_score)
        confidence_factors['semantic'] = min(semantic_score * 40, 40)
        
        # 2. Symptom keyword matching (30% weight)
        user_input_lower = user_input.lower()
        searchable_text = f"{code_data.get('description', '')} {' '.join(code_data.get('common_causes', []))}"
        
        # Count exact keyword matches
        keyword_matches = 0
        total_keywords = 0
        
        for symptom_type, keywords in self.symptom_keywords.items():
            total_keywords += len(keywords)
            for keyword in keywords:
                if keyword in user_input_lower and keyword in searchable_text.lower():
                    keyword_matches += 1
        
        keyword_score = (keyword_matches / max(total_keywords, 1)) * 30 if total_keywords > 0 else 0
        confidence_factors['keywords'] = min(keyword_score, 30)
        
        # 3. Automotive context strength (15% weight)
        automotive_terms = [
            'engine', 'motor', 'car', 'vehicle', 'check engine', 'light', 'code',
            'misfire', 'idle', 'rough', 'stall', 'oxygen', 'sensor', 'fuel',
            'exhaust', 'emission', 'coolant', 'temperature', 'brake', 'transmission'
        ]
        
        context_matches = sum(1 for term in automotive_terms if term in user_input_lower)
        context_score = min(context_matches / len(automotive_terms), 1.0) * 15
        confidence_factors['context'] = context_score
        
        # 4. Input specificity (10% weight)
        tokens = user_input.split()
        specificity_score = min(len(tokens) / 10, 1.0) * 10  # More words = higher specificity
        confidence_factors['specificity'] = specificity_score
        
        # 5. Code priority adjustment (5% weight)
        priority_map = {'High': 5, 'Medium': 3, 'Low': 1}
        priority_boost = priority_map.get(code_data.get('priority', 'Medium'), 3)
        confidence_factors['priority'] = priority_boost
        
        # Calculate final confidence
        total_confidence = sum(confidence_factors.values())
        
        # Apply penalties for certain conditions
        penalties = 0
        
        # Penalty for very short input
        if len(user_input.split()) < 3:
            penalties += 10
            
        # Penalty for generic terms
        generic_terms = ['problem', 'issue', 'help', 'fix', 'broken']
        if any(term in user_input_lower for term in generic_terms) and len(tokens) < 5:
            penalties += 5
        
        final_confidence = max(total_confidence - penalties, 0)
        
        return min(final_confidence, 100), confidence_factors

    def _validate_diagnostic_relevance(self, user_input):
        """
        Enhanced validation to determine if input is vehicle diagnostic related
        Returns (is_valid, confidence_score, reason)
        """
        if not user_input or not user_input.strip():
            return False, 0, "Empty input provided"
        
        user_input_lower = user_input.lower().strip()
        tokens = user_input.split()
        
        # Check for non-diagnostic phrases
        non_diagnostic = ['hi', 'hello', 'hey', 'thanks', 'thank you', 'ok', 'okay']
        if user_input_lower in non_diagnostic:
            return False, 0, "Input appears to be a greeting or general response"
        
        # Calculate diagnostic relevance confidence
        diagnostic_confidence = 0
        
        # 1. Automotive keyword presence (50% weight)
        automotive_keywords = set()
        for symptom_list in self.symptom_keywords.values():
            automotive_keywords.update(symptom_list)
        
        automotive_keywords.update([
            'car', 'vehicle', 'engine', 'motor', 'transmission', 'brake', 'wheel', 
            'tire', 'battery', 'starter', 'alternator', 'radiator', 'exhaust',
            'oil', 'coolant', 'fluid', 'light', 'dashboard', 'check engine',
            'rpm', 'idle', 'acceleration', 'speed', 'gear', 'clutch', 'code'
        ])
        
        automotive_matches = sum(1 for keyword in automotive_keywords if keyword in user_input_lower)
        auto_score = min(automotive_matches / 3, 1.0) * 50  # Up to 50 points
        diagnostic_confidence += auto_score
        
        # 2. Problem indicators (25% weight)
        problem_indicators = [
            'problem', 'issue', 'trouble', 'error', 'fault', 'wrong', 'broken',
            'not working', 'wont', 'can\'t', 'difficult', 'hard', 'strange',
            'weird', 'unusual', 'noise', 'sound', 'vibration', 'smoke', 'leak'
        ]
        
        problem_matches = sum(1 for indicator in problem_indicators if indicator in user_input_lower)
        problem_score = min(problem_matches / 2, 1.0) * 25  # Up to 25 points
        diagnostic_confidence += problem_score
        
        # 3. Input length and specificity (15% weight)
        length_score = min(len(tokens) / 8, 1.0) * 15  # Up to 15 points
        diagnostic_confidence += length_score
        
        # 4. Technical terms presence (10% weight)
        technical_terms = ['sensor', 'code', 'dtc', 'obd', 'ecu', 'pcm', 'abs', 'airbag']
        tech_matches = sum(1 for term in technical_terms if term in user_input_lower)
        tech_score = min(tech_matches, 1.0) * 10  # Up to 10 points
        diagnostic_confidence += tech_score
        
        # Determine if valid based on confidence threshold
        is_valid = diagnostic_confidence >= 30  # 30% minimum for diagnostic relevance
        
        if not is_valid:
            return False, diagnostic_confidence, f"Input doesn't appear to be vehicle diagnostic related (confidence: {diagnostic_confidence:.1f}%)"
        
        return True, diagnostic_confidence, f"Valid diagnostic input (confidence: {diagnostic_confidence:.1f}%)"
    
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
                if similarity > 0.1:  # Minimum threshold for relevance
                    results.append({
                        'code_id': code_ids[i],
                        'similarity': similarity,
                        'method': 'tfidf'
                    })
            
            return sorted(results, key=lambda x: x['similarity'], reverse=True)
            
        except Exception as e:
            print(f"Error calculating TF-IDF similarity: {e}")
            return []
    
    def _calculate_fuzzy_similarity(self, user_input):
        """Calculate fuzzy string similarity between user input and OBD codes"""
        results = []
        user_input_lower = user_input.lower()
        
        # Skip if input is too short for meaningful fuzzy matching
        if len(user_input_lower) < 3:
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
                partial_score * 0.5 +
                token_sort_score * 0.2 + 
                token_set_score * 0.1
            )
            
            # More lenient threshold for fuzzy matching
            if weighted_score > 0.25:
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
    
    def process_user_input(self, user_input, top_n=5, confidence_threshold=70.0):
        """
        Enhanced processing with confidence threshold for LLM routing
        
        Args:
            user_input (str): User's description of the vehicle problem
            top_n (int): Number of top results to return
            confidence_threshold (float): Minimum confidence to pass codes to LLM
            
        Returns:
            dict: Contains matched codes, confidence scores, routing decision, and analysis
        """
        # Enhanced validation with diagnostic relevance check
        is_valid, diagnostic_confidence, validation_message = self._validate_diagnostic_relevance(user_input)
        
        if not is_valid:
            return {
                'success': False,
                'route_to_llm': True,  # Route non-diagnostic queries to LLM
                'diagnostic_confidence': diagnostic_confidence,
                'error': validation_message,
                'matches': [],
                'analysis': {
                    'validation_failed': True,
                    'reason': validation_message,
                    'diagnostic_relevance': diagnostic_confidence
                }
            }
        
        # Extract symptoms
        symptoms = self._extract_symptoms(user_input)
        
        # Calculate similarities using different methods
        tfidf_results = self._calculate_tfidf_similarity(user_input)
        fuzzy_results = self._calculate_fuzzy_similarity(user_input)
        
        # If no meaningful results from either method
        if not tfidf_results and not fuzzy_results:
            return {
                'success': False,
                'route_to_llm': True,  # Route to LLM when no codes found
                'diagnostic_confidence': diagnostic_confidence,
                'error': 'No matching diagnostic codes found in database.',
                'matches': [],
                'analysis': {
                    'detected_symptoms': symptoms,
                    'total_codes_analyzed': len(self.obd_codes),
                    'tfidf_matches': 0,
                    'fuzzy_matches': 0,
                    'user_input_processed': self._preprocess_text(user_input),
                    'diagnostic_relevance': diagnostic_confidence
                }
            }
        
        # Enhanced ranking with confidence calculation  
        processed_codes = {}
        
        # Process TF-IDF results
        for result in tfidf_results[:10]:
            code_id = result['code_id']
            code_data = self._find_code_by_id(code_id)
            if code_data:
                confidence, factors = self._calculate_enhanced_confidence(
                    user_input, code_data, 
                    tfidf_score=result['similarity'], 
                    symptoms_matched=symptoms
                )
                processed_codes[code_id] = {
                    'confidence': confidence,
                    'factors': factors,
                    'tfidf_score': result['similarity'],
                    'fuzzy_score': 0
                }
        
        # Process fuzzy results (merge with existing or add new)
        for result in fuzzy_results[:10]:
            code_id = result['code_id']
            code_data = self._find_code_by_id(code_id)
            if code_data:
                if code_id in processed_codes:
                    # Update with fuzzy score and recalculate
                    processed_codes[code_id]['fuzzy_score'] = result['similarity']
                    confidence, factors = self._calculate_enhanced_confidence(
                        user_input, code_data,
                        tfidf_score=processed_codes[code_id]['tfidf_score'],
                        fuzzy_score=result['similarity'],
                        symptoms_matched=symptoms
                    )
                    processed_codes[code_id]['confidence'] = confidence
                    processed_codes[code_id]['factors'] = factors
                else:
                    # New code from fuzzy matching
                    confidence, factors = self._calculate_enhanced_confidence(
                        user_input, code_data,
                        fuzzy_score=result['similarity'],
                        symptoms_matched=symptoms
                    )
                    processed_codes[code_id] = {
                        'confidence': confidence,
                        'factors': factors,
                        'tfidf_score': 0,
                        'fuzzy_score': result['similarity']
                    }
        
        # Filter and sort by confidence
        min_confidence = 20.0  # 20% minimum confidence
        filtered_results = [
            (code_id, data) for code_id, data in processed_codes.items()
            if data['confidence'] >= min_confidence
        ]
        enhanced_results = sorted(filtered_results, key=lambda x: x[1]['confidence'], reverse=True)
        
        # Check if any results meet the confidence threshold
        high_confidence_matches = [
            (code_id, data) for code_id, data in enhanced_results 
            if data['confidence'] >= confidence_threshold
        ]
        
        # Determine routing decision
        route_to_llm = len(high_confidence_matches) == 0
        max_confidence = max([data['confidence'] for _, data in enhanced_results], default=0)
        
        # Prepare results
        matches = []
        results_to_process = enhanced_results[:top_n] if enhanced_results else []
        
        for code_id, confidence_data in results_to_process:
            code_data = self._find_code_by_id(code_id)
            if code_data:
                matches.append({
                    'code_id': code_data['id'],
                    'description': code_data['description'],
                    'common_causes': code_data.get('common_causes', []),
                    'priority': code_data.get('priority', 'Medium'),
                    'confidence_score': confidence_data['confidence'] / 100.0,  # Convert to 0-1 scale
                    'confidence_percentage': f"{confidence_data['confidence']:.1f}%",
                    'confidence_factors': confidence_data['factors'],
                    'confirmation': code_data.get('likely_cause', '')
                })
        
        return {
            'success': True,
            'route_to_llm': route_to_llm,
            'diagnostic_confidence': diagnostic_confidence,
            'max_confidence': max_confidence,
            'high_confidence_count': len(high_confidence_matches),
            'message': f"Found {len(matches)} matching diagnostic codes" + 
                      (f" (max confidence: {max_confidence:.1f}%)" if matches else ""),
            'matches': matches,
            'analysis': {
                'detected_symptoms': symptoms,
                'total_codes_analyzed': len(self.obd_codes),
                'tfidf_matches': len(tfidf_results),
                'fuzzy_matches': len(fuzzy_results),
                'enhanced_results_count': len(enhanced_results),
                'confidence_threshold': confidence_threshold,
                'routing_decision': 'NLP' if not route_to_llm else 'LLM',
                'diagnostic_relevance': diagnostic_confidence,
                'user_input_processed': self._preprocess_text(user_input)
            }
        }


def main():
    """Test the enhanced diagnostic processor"""
    # Initialize processor
    obd_codes_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'obd_codes.json')
    processor = EnhancedDiagnosticProcessor(obd_codes_path)
    
    # Test with sample inputs
    test_inputs = [
        "My car has a rough idle and engine misfire",
        "Check engine light is on, oxygen sensor issue", 
        "How do I change my oil?",  # Non-diagnostic query
        "Hi there",  # Greeting
        "Engine is running cold, thermostat problem",
        "What's the weather like?",  # Completely unrelated
    ]
    
    print("=== Enhanced Vehicle Diagnostic NLP Processor Test ===\n")
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"Test {i}: '{test_input}'")
        results = processor.process_user_input(test_input, top_n=3, confidence_threshold=70.0)
        
        print(f"Route to LLM: {results.get('route_to_llm', False)}")
        print(f"Diagnostic Confidence: {results.get('diagnostic_confidence', 0):.1f}%")
        print(f"Max Match Confidence: {results.get('max_confidence', 0):.1f}%")
        
        if results.get('matches'):
            print("Top matches:")
            for j, match in enumerate(results['matches'], 1):
                print(f"  {j}. {match['code_id']}: {match['description']}")
                print(f"     Priority: {match['priority']}, Confidence: {match['confidence_percentage']}")
        else:
            print("No matches found")
        
        print("-" * 50)


if __name__ == "__main__":
    main()