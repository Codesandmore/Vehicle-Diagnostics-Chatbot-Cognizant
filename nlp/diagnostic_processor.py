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
    
    def _calculate_enhanced_confidence(self, user_input, code_data, tfidf_score=0, fuzzy_score=0, symptoms_matched=[]):
        """
        AGGRESSIVE confidence calculation using ACTUAL OBD data comparison
        Now properly compares user input with actual JSON data content
        """
        factors = {}
        user_lower = user_input.lower().strip()
        
        # Get ACTUAL text content from the OBD code data
        code_description = code_data.get('description', '').lower()
        code_causes = [cause.lower() for cause in code_data.get('common_causes', [])]
        code_id = code_data.get('id', '').lower()
        
        print(f"🔍 CONFIDENCE: Analyzing {code_id.upper()} vs '{user_input}'")
        print(f"   Code description: '{code_description}'")
        print(f"   Code causes: {code_causes}")
        
        # 1. EXACT CODE ID MATCH (Instant 95%)
        if code_id and code_id in user_lower:
            print(f"🎯 EXACT CODE ID MATCH: {code_id.upper()} found in user input!")
            return 95, {'exact_code_id': True, 'instant_match': True}
        
        # 2. DIRECT TEXT MATCHING WITH ACTUAL DATA (60% weight)
        text_match_score = 0
        
        # A. EXACT DESCRIPTION MATCHING
        # Split description into meaningful words
        desc_words = [word for word in code_description.split() if len(word) > 2]
        matched_desc_words = []
        
        for word in desc_words:
            if word in user_lower:
                matched_desc_words.append(word)
                text_match_score += 8  # 8 points per description word match
                print(f"   ✅ Description word match: '{word}'")
        
        # B. EXACT PHRASE MATCHING from description
        # Check for multi-word phrases from actual description
        if len(desc_words) > 1:
            for i in range(len(desc_words) - 1):
                for length in [3, 2]:  # Try 3-word and 2-word phrases
                    if i + length <= len(desc_words):
                        phrase = ' '.join(desc_words[i:i+length])
                        if len(phrase) > 6 and phrase in user_lower:
                            text_match_score += 20  # Big bonus for phrase matches
                            print(f"   🎯 EXACT PHRASE MATCH: '{phrase}' (+20)")
        
        # C. COMMON CAUSES MATCHING with actual data
        matched_causes = []
        for cause in code_causes:
            cause_words = [word for word in cause.split() if len(word) > 2]
            
            # Check for exact cause matches
            if cause in user_lower:
                text_match_score += 15  # 15 points for exact cause match
                matched_causes.append(cause)
                print(f"   🎯 EXACT CAUSE MATCH: '{cause}' (+15)")
            else:
                # Check for partial cause word matches
                cause_word_matches = 0
                for word in cause_words:
                    if word in user_lower:
                        cause_word_matches += 1
                
                if cause_word_matches > 0:
                    partial_score = (cause_word_matches / len(cause_words)) * 10
                    text_match_score += partial_score
                    print(f"   ✅ Partial cause match: '{cause}' - {cause_word_matches}/{len(cause_words)} words (+{partial_score:.1f})")
        
        text_match_score = min(text_match_score, 60)  # Cap at 60%
        factors['actual_data_matching'] = text_match_score
        
        # 3. SEMANTIC SIMILARITY FROM NLP (25% weight)
        semantic_score = max(tfidf_score, fuzzy_score) * 25
        factors['nlp_semantic_score'] = semantic_score
        
        # 4. WORD OVERLAP WITH ACTUAL DATA (10% weight)
        all_code_words = set()
        all_code_words.update(desc_words)
        for cause in code_causes:
            all_code_words.update(cause.split())
        
        user_words = set([word for word in user_lower.split() if len(word) > 2])
        
        if all_code_words and user_words:
            word_overlap = all_code_words.intersection(user_words)
            overlap_ratio = len(word_overlap) / len(all_code_words)
            overlap_score = overlap_ratio * 10
            
            if overlap_ratio > 0.3:  # 30%+ overlap gets bonus
                overlap_score += 5
                print(f"   🎯 HIGH WORD OVERLAP: {overlap_ratio:.1%} (+5 bonus)")
            
            factors['word_overlap_actual'] = overlap_score
            print(f"   Word overlap: {word_overlap}")
        else:
            factors['word_overlap_actual'] = 0
        
        # 5. AUTOMOTIVE CONTEXT (5% weight)
        automotive_bonus = 0
        automotive_indicators = ['sensor', 'engine', 'system', 'circuit', 'malfunction', 'fault', 'code', 'dtc', 'obd']
        for indicator in automotive_indicators:
            if indicator in user_lower:
                automotive_bonus += 1
        
        automotive_score = min(automotive_bonus, 5)
        factors['automotive_context'] = automotive_score
        
        # Calculate base confidence
        base_confidence = sum(factors.values())
        
        # 6. AGGRESSIVE MULTIPLIERS based on actual data matches
        final_confidence = base_confidence
        
        # SUPER BOOST: Strong matches across multiple factors
        strong_factors = sum([
            1 if text_match_score >= 30 else 0,  # Strong text matching
            1 if semantic_score >= 15 else 0,    # Strong semantic similarity
            1 if len(matched_desc_words) >= 2 else 0,  # Multiple description words
            1 if len(matched_causes) >= 1 else 0      # At least one cause match
        ])
        
        if strong_factors >= 3:  # Multiple strong indicators
            final_confidence = min(final_confidence * 1.4, 96)  # 40% boost!
            factors['multi_strong_boost'] = True
            print(f"🚀 MULTI-STRONG BOOST: {strong_factors} strong factors! (+40%)")
        elif strong_factors >= 2:  # Two strong indicators
            final_confidence = min(final_confidence * 1.25, 94)  # 25% boost
            factors['double_strong_boost'] = True
            print(f"🚀 DOUBLE-STRONG BOOST: {strong_factors} strong factors (+25%)")
        elif strong_factors >= 1:  # One strong indicator
            final_confidence = min(final_confidence * 1.1, 92)  # 10% boost
            factors['single_strong_boost'] = True
            print(f"🎯 STRONG BOOST: {strong_factors} strong factor (+10%)")
        
        # SPECIAL CASES based on actual data
        # Perfect description match
        if len(matched_desc_words) >= len(desc_words) and len(desc_words) >= 2:
            final_confidence = min(final_confidence + 15, 96)
            factors['perfect_description_match'] = True
            print(f"🎯 PERFECT DESCRIPTION MATCH! (+15)")
        
        # Exact cause match
        if matched_causes:
            final_confidence = min(final_confidence + 10, 96)
            factors['exact_cause_match'] = True
            print(f"🎯 EXACT CAUSE MATCH! (+10)")
        
        # MINIMUM THRESHOLDS
        if text_match_score == 0 and semantic_score < 5:
            final_confidence = min(final_confidence, 25)  # Cap very weak matches
            factors['weak_match_penalty'] = True
            print(f"⚠️  WEAK MATCH: No text matches and low semantic score")
        
        # Final confidence (cap at 96% to maintain uncertainty)
        final_confidence = min(max(final_confidence, 0), 96)
        
        print(f"   📊 FINAL CONFIDENCE: {final_confidence:.1f}%")
        print(f"   📊 Breakdown: Text={text_match_score:.1f}, Semantic={semantic_score:.1f}, Overlap={factors.get('word_overlap_actual', 0):.1f}")
        
        return final_confidence, factors
    
    def _assess_diagnostic_relevance(self, user_input):
        """
        Assess how diagnostically relevant the input is using COMPREHENSIVE automotive vocabulary
        Returns relevance score (0-100) and classification
        """
        user_lower = user_input.lower().strip()
        tokens = [token.strip() for token in user_input.split()]
        
        print(f"🔍 RELEVANCE: Assessing '{user_input}'")
        
        # Check for obvious non-diagnostic content first (use word boundaries)
        non_diagnostic_patterns = [
            'hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye',
            'weather', 'time', 'date', 'news', 'music', 'recipe',
            'how are you', 'what\'s up', 'good morning', 'good evening'
        ]
        
        # Split user input into words for exact matching
        user_words = set(user_lower.split())
        
        for pattern in non_diagnostic_patterns:
            # Check for exact word matches or phrase matches
            if ' ' in pattern:  # Multi-word patterns
                if pattern in user_lower:
                    print(f"   ❌ Non-diagnostic phrase detected: '{pattern}'")
                    return 5, "non_diagnostic"
            else:  # Single word patterns
                if pattern in user_words:
                    print(f"   ❌ Non-diagnostic word detected: '{pattern}'")
                    return 5, "non_diagnostic"
        
        relevance_score = 0
        matched_categories = []
        
        # 1. COMPREHENSIVE AUTOMOTIVE VOCABULARY (50 points)
        # Use actual automotive terms from the OBD codes data
        automotive_vocab = {
            # Basic vehicle components
            'car', 'vehicle', 'engine', 'motor', 'transmission', 'brake', 'brakes', 'wheel', 'wheels',
            'battery', 'oil', 'fuel', 'gas', 'gasoline', 'diesel', 'coolant', 'radiator', 'alternator',
            'starter', 'exhaust', 'muffler', 'tire', 'tires', 'steering', 'suspension',
            
            # Engine and powertrain
            'cylinder', 'piston', 'valve', 'valves', 'camshaft', 'crankshaft', 'timing', 'belt',
            'chain', 'head', 'block', 'manifold', 'intake', 'throttle', 'carburetor',
            
            # Electrical and sensors
            'sensor', 'sensors', 'oxygen', 'o2', 'lambda', 'maf', 'map', 'tps', 'iac', 'cam', 'crank',
            'coil', 'coils', 'plug', 'plugs', 'wire', 'wires', 'harness', 'connector',
            
            # Fuel system
            'injector', 'injectors', 'pump', 'filter', 'rail', 'pressure', 'regulator',
            
            # Transmission and drivetrain
            'shift', 'shifting', 'gear', 'gears', 'clutch', 'torque', 'converter', 'differential',
            'axle', 'driveshaft', 'cv', 'joint', 'solenoid', 'solenoids',
            
            # Exhaust and emissions
            'catalytic', 'catalyst', 'converter', 'evap', 'egr', 'pcv', 'emission', 'emissions',
            
            # Cooling system
            'thermostat', 'hose', 'hoses', 'water', 'antifreeze', 'fan', 'temperature',
            
            # Braking system
            'pad', 'pads', 'rotor', 'rotors', 'caliper', 'master', 'booster', 'abs',
            
            # Electrical system
            'fuse', 'relay', 'switch', 'motor', 'actuator', 'module', 'ecu', 'pcm', 'bcm',
            
            # Diagnostic terms
            'code', 'codes', 'dtc', 'obd', 'check', 'light', 'warning', 'dash', 'dashboard',
            'scanner', 'diagnostic', 'diagnostics', 'scan', 'reader'
        }
        
        auto_matches = []
        for word in automotive_vocab:
            if word in user_lower:
                auto_matches.append(word)
        
        auto_score = min(len(auto_matches) * 8, 50)  # 8 points per match, max 50
        relevance_score += auto_score
        
        if auto_matches:
            matched_categories.append(f"automotive_terms({len(auto_matches)})")
            print(f"   ✅ Automotive terms: {auto_matches}")
        
        # 2. PROBLEM/SYMPTOM INDICATORS (30 points)
        problem_indicators = {
            # Direct problem words
            'problem', 'issue', 'issues', 'trouble', 'error', 'fault', 'faulty', 'defective',
            'broken', 'bad', 'failed', 'failing', 'malfunction', 'malfunctioning',
            
            # Negative indicators
            'not working', 'won\'t', 'wont', 'can\'t', 'cant', 'doesn\'t', 'doesnt',
            'refuse', 'refuses', 'stopped', 'quit', 'dead',
            
            # Performance issues
            'rough', 'hard', 'difficult', 'poor', 'low', 'high', 'strange', 'weird', 'unusual',
            'intermittent', 'occasional', 'sometimes', 'hesitation', 'hesitating', 'stalling',
            'jerking', 'shaking', 'vibrating', 'grinding', 'squealing', 'knocking',
            
            # Operational issues
            'slow', 'fast', 'loud', 'noisy', 'smoking', 'overheating', 'leaking', 'leak'
        }
        
        problem_matches = []
        for indicator in problem_indicators:
            if indicator in user_lower:
                problem_matches.append(indicator)
        
        problem_score = min(len(problem_matches) * 6, 30)  # 6 points per match, max 30
        relevance_score += problem_score
        
        if problem_matches:
            matched_categories.append(f"problems({len(problem_matches)})")
            print(f"   ✅ Problem indicators: {problem_matches}")
        
        # 3. TECHNICAL/DIAGNOSTIC TERMS (20 points)
        tech_terms = {
            'misfire', 'misfiring', 'lean', 'rich', 'knock', 'ping', 'surge', 'idle', 'rpm',
            'voltage', 'resistance', 'current', 'ground', 'short', 'open', 'circuit',
            'bank', 'sensor', 'heater', 'catalyst', 'efficiency', 'monitor', 'readiness',
            'pending', 'stored', 'freeze', 'frame', 'data', 'pid', 'parameter'
        }
        
        tech_matches = []
        for term in tech_terms:
            if term in user_lower:
                tech_matches.append(term)
        
        tech_score = min(len(tech_matches) * 5, 20)  # 5 points per match, max 20
        relevance_score += tech_score
        
        if tech_matches:
            matched_categories.append(f"technical({len(tech_matches)})")
            print(f"   ✅ Technical terms: {tech_matches}")
        
        # CLASSIFY BASED ON SCORE
        if relevance_score >= 70:
            classification = "highly_diagnostic"
        elif relevance_score >= 40:
            classification = "moderately_diagnostic"  
        elif relevance_score >= 20:
            classification = "possibly_diagnostic"
        else:
            classification = "non_diagnostic"
        
        print(f"   📊 RELEVANCE SCORE: {relevance_score}/100 ({classification})")
        print(f"   📊 Matched categories: {', '.join(matched_categories) if matched_categories else 'None'}")
        
        return relevance_score, classification

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
        """
        Calculate TF-IDF cosine similarity between user input and ALL OBD codes data
        Now properly scans the actual JSON data instead of predefined keywords
        """
        if self.tfidf_matrix is None:
            return []
            
        # Preprocess user input
        processed_input = self._preprocess_text(user_input)
        if not processed_input.strip():
            return []
        
        print(f"🔍 TF-IDF: Scanning ALL {len(self.obd_codes)} OBD codes for: '{user_input}'")
            
        try:
            # Transform user input using fitted vectorizer
            user_vector = self.vectorizer.transform([processed_input])
            
            # Calculate cosine similarity against ALL codes
            similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
            
            # Create results list with detailed matching info
            results = []
            code_ids = list(self.searchable_texts.keys())
            
            for i, similarity in enumerate(similarities):
                if similarity > 0.02:  # Very low threshold to catch more potential matches
                    code_id = code_ids[i]
                    
                    # Find the actual code data for this match
                    code_data = self._find_code_by_id(code_id)
                    if code_data:
                        matched_text = self.searchable_texts[code_id]
                        
                        results.append({
                            'code_id': code_id,
                            'similarity': float(similarity),
                            'method': 'tfidf',
                            'description': code_data['description'],
                            'matched_content': matched_text[:100] + "..." if len(matched_text) > 100 else matched_text
                        })
                        
                        # Log significant matches
                        if similarity > 0.1:
                            print(f"   🎯 TF-IDF Strong Match: {code_id} ({similarity:.3f}) - {code_data['description']}")
                        elif similarity > 0.05:
                            print(f"   ✅ TF-IDF Good Match: {code_id} ({similarity:.3f}) - {code_data['description']}")
            
            # Sort by similarity score (highest first)
            sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)
            
            print(f"🔍 TF-IDF: Found {len(sorted_results)} potential matches")
            if sorted_results:
                print(f"   Top match: {sorted_results[0]['code_id']} ({sorted_results[0]['similarity']:.3f})")
                
            return sorted_results
            
        except Exception as e:
            print(f"❌ Error in TF-IDF calculation: {e}")
            return []
    
    def _calculate_fuzzy_similarity(self, user_input):
        """
        Calculate fuzzy string similarity for exact matches against ALL actual data
        Now properly scans each OBD code's description and causes individually
        """
        results = []
        user_input_lower = user_input.lower().strip()
        
        # Skip if input is too short for meaningful fuzzy matching
        if len(user_input_lower) < 3:
            return []
        
        print(f"🔍 FUZZY: Scanning ALL {len(self.obd_codes)} OBD codes for: '{user_input}'")
        
        # Scan EVERY single OBD code in the actual data
        for code_data in self.obd_codes:
            code_id = code_data['id']
            
            # Get all the text content from this code
            description = code_data.get('description', '').lower()
            common_causes = [cause.lower() for cause in code_data.get('common_causes', [])]
            all_causes_text = ' '.join(common_causes)
            
            # Calculate fuzzy scores against different parts of the code data
            max_score = 0
            best_match_text = ""
            match_type = ""
            
            # 1. Match against description
            if description:
                desc_ratio = fuzz.ratio(user_input_lower, description) / 100.0
                desc_partial = fuzz.partial_ratio(user_input_lower, description) / 100.0
                desc_token_sort = fuzz.token_sort_ratio(user_input_lower, description) / 100.0
                desc_token_set = fuzz.token_set_ratio(user_input_lower, description) / 100.0
                
                # Weighted score for description (prioritize partial and token matches)
                desc_score = (desc_ratio * 0.2 + desc_partial * 0.4 + desc_token_sort * 0.3 + desc_token_set * 0.1)
                
                if desc_score > max_score:
                    max_score = desc_score
                    best_match_text = description
                    match_type = "description"
            
            # 2. Match against common causes combined
            if all_causes_text:
                causes_ratio = fuzz.ratio(user_input_lower, all_causes_text) / 100.0
                causes_partial = fuzz.partial_ratio(user_input_lower, all_causes_text) / 100.0
                causes_token_sort = fuzz.token_sort_ratio(user_input_lower, all_causes_text) / 100.0
                causes_token_set = fuzz.token_set_ratio(user_input_lower, all_causes_text) / 100.0
                
                # Weighted score for causes
                causes_score = (causes_ratio * 0.1 + causes_partial * 0.5 + causes_token_sort * 0.3 + causes_token_set * 0.1)
                
                if causes_score > max_score:
                    max_score = causes_score
                    best_match_text = all_causes_text[:50] + "..."
                    match_type = "causes"
            
            # 3. Match against individual causes (for precise matching)
            for cause in common_causes:
                if cause and len(cause) > 5:  # Skip very short causes
                    cause_ratio = fuzz.ratio(user_input_lower, cause) / 100.0
                    cause_partial = fuzz.partial_ratio(user_input_lower, cause) / 100.0
                    cause_token_sort = fuzz.token_sort_ratio(user_input_lower, cause) / 100.0
                    
                    # Individual cause score
                    individual_cause_score = (cause_ratio * 0.3 + cause_partial * 0.4 + cause_token_sort * 0.3)
                    
                    if individual_cause_score > max_score:
                        max_score = individual_cause_score
                        best_match_text = cause
                        match_type = "individual_cause"
            
            # Include if score is above threshold (lowered for better coverage)
            if max_score > 0.2:  # 20% threshold instead of 25%
                results.append({
                    'code_id': code_id,
                    'similarity': max_score,
                    'method': 'fuzzy',
                    'description': code_data['description'],
                    'match_type': match_type,
                    'matched_text': best_match_text[:50] + "..." if len(best_match_text) > 50 else best_match_text
                })
                
                # Log significant matches
                if max_score > 0.7:
                    print(f"   🎯 FUZZY Excellent Match: {code_id} ({max_score:.3f}) - {match_type}: '{best_match_text[:30]}...'")
                elif max_score > 0.5:
                    print(f"   ✅ FUZZY Good Match: {code_id} ({max_score:.3f}) - {match_type}: '{best_match_text[:30]}...'")
                elif max_score > 0.3:
                    print(f"   🔍 FUZZY Potential Match: {code_id} ({max_score:.3f}) - {match_type}")
        
        # Sort by similarity score (highest first)
        sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        
        print(f"🔍 FUZZY: Found {len(sorted_results)} potential matches")
        if sorted_results:
            top_match = sorted_results[0]
            print(f"   Top match: {top_match['code_id']} ({top_match['similarity']:.3f}) - {top_match['match_type']}")
        
        return sorted_results
    
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
    
    def process_user_input(self, user_input, top_n=5, confidence_threshold=50.0):
        """
        Enhanced processing with 90% confidence threshold for intelligent LLM routing
        
        Args:
            user_input (str): User's description of the vehicle problem
            top_n (int): Number of top results to return
            confidence_threshold (float): Minimum confidence to pass codes to LLM (default 90%)
            
        Returns:
            dict: Contains matched codes, routing decision, confidence scores, and analysis
        """
        # First assess diagnostic relevance
        relevance_score, relevance_class = self._assess_diagnostic_relevance(user_input)
        
        # If clearly non-diagnostic, route directly to LLM
        if relevance_class == "non_diagnostic":
            return {
                'success': False,
                'route_to_llm_only': True,
                'diagnostic_relevance': relevance_score,
                'relevance_classification': relevance_class,
                'reason': 'Query appears to be non-diagnostic - routing to general LLM',
                'matches': [],
                'analysis': {
                    'diagnostic_relevance_score': relevance_score,
                    'classification': relevance_class,
                    'routing_decision': 'LLM_ONLY'
                }
            }
        
        # Validate diagnostic input
        is_valid, validation_message = self._is_valid_diagnostic_input(user_input)
        if not is_valid:
            return {
                'success': False,
                'route_to_llm_only': True,
                'diagnostic_relevance': relevance_score,
                'error': validation_message,
                'matches': [],
                'analysis': {
                    'validation_failed': True,
                    'reason': validation_message,
                    'diagnostic_relevance_score': relevance_score,
                    'routing_decision': 'LLM_ONLY'
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
                'route_to_llm_only': True,
                'diagnostic_relevance': relevance_score,
                'reason': 'No diagnostic codes found in database - routing to LLM for general advice',
                'matches': [],
                'analysis': {
                    'detected_symptoms': symptoms,
                    'total_codes_analyzed': len(self.obd_codes),
                    'tfidf_matches': 0,
                    'fuzzy_matches': 0,
                    'diagnostic_relevance_score': relevance_score,
                    'routing_decision': 'LLM_ONLY'
                }
            }
        
        # Enhanced confidence calculation for each potential match
        enhanced_matches = []
        
        # Process TF-IDF results
        for result in tfidf_results[:15]:  # Process more results for better confidence assessment
            code_data = self._find_code_by_id(result['code_id'])
            if code_data:
                confidence, confidence_details = self._calculate_enhanced_confidence(
                    user_input, code_data, 
                    tfidf_score=result['similarity'],
                    symptoms_matched=symptoms
                )
                
                enhanced_matches.append({
                    'code_id': result['code_id'],
                    'code_data': code_data,
                    'confidence': confidence,
                    'confidence_details': confidence_details,
                    'source': 'tfidf'
                })
        
        # Process fuzzy results and merge/add
        processed_codes = {match['code_id']: match for match in enhanced_matches}
        
        for result in fuzzy_results[:15]:
            code_id = result['code_id']
            code_data = self._find_code_by_id(code_id)
            
            if code_data:
                if code_id in processed_codes:
                    # Recalculate with both scores
                    existing_match = processed_codes[code_id]
                    tfidf_score = next((r['similarity'] for r in tfidf_results if r['code_id'] == code_id), 0)
                    
                    confidence, confidence_details = self._calculate_enhanced_confidence(
                        user_input, code_data,
                        tfidf_score=tfidf_score,
                        fuzzy_score=result['similarity'],
                        symptoms_matched=symptoms
                    )
                    
                    processed_codes[code_id].update({
                        'confidence': confidence,
                        'confidence_details': confidence_details,
                        'source': 'tfidf+fuzzy'
                    })
                else:
                    # New match from fuzzy only
                    confidence, confidence_details = self._calculate_enhanced_confidence(
                        user_input, code_data,
                        fuzzy_score=result['similarity'],
                        symptoms_matched=symptoms
                    )
                    
                    processed_codes[code_id] = {
                        'code_id': code_id,
                        'code_data': code_data,
                        'confidence': confidence,
                        'confidence_details': confidence_details,
                        'source': 'fuzzy'
                    }
        
        # Sort by confidence and filter minimum threshold
        sorted_matches = sorted(processed_codes.values(), key=lambda x: x['confidence'], reverse=True)
        filtered_matches = [match for match in sorted_matches if match['confidence'] >= 20.0]  # 20% minimum
        
        # Determine routing based on confidence threshold
        high_confidence_matches = [match for match in filtered_matches if match['confidence'] >= confidence_threshold]
        max_confidence = max([match['confidence'] for match in filtered_matches], default=0)
        
        # Routing decision logic
        if high_confidence_matches:
            routing_decision = 'NLP_TO_LLM'  # Pass codes to LLM for detailed explanation
            route_to_llm_only = False
        else:
            routing_decision = 'LLM_ONLY'    # Skip codes, go directly to LLM
            route_to_llm_only = True
        
        # Prepare final matches for output
        final_matches = []
        matches_to_process = filtered_matches[:top_n]
        
        for match in matches_to_process:
            code_data = match['code_data']
            final_matches.append({
                'code_id': code_data['id'],
                'description': code_data['description'],
                'common_causes': code_data.get('common_causes', []),
                'priority': code_data.get('priority', 'Medium'),
                'confidence_score': match['confidence'] / 100.0,  # 0-1 scale for compatibility
                'confidence_percentage': f"{match['confidence']:.1f}%",
                'confidence_breakdown': match['confidence_details'],
                'confirmation': code_data.get('likely_cause', ''),
                'source_method': match['source']
            })
        
        # Create comprehensive response
        return {
            'success': True,
            'route_to_llm_only': route_to_llm_only,
            'routing_decision': routing_decision,
            'diagnostic_relevance': relevance_score,
            'relevance_classification': relevance_class,
            'max_confidence': max_confidence,
            'high_confidence_count': len(high_confidence_matches),
            'confidence_threshold': confidence_threshold,
            'message': self._generate_response_message(final_matches, routing_decision, max_confidence),
            'matches': final_matches,
            'analysis': {
                'detected_symptoms': symptoms,
                'total_codes_analyzed': len(self.obd_codes),
                'tfidf_matches': len(tfidf_results),
                'fuzzy_matches': len(fuzzy_results),
                'enhanced_matches': len(filtered_matches),
                'high_confidence_matches': len(high_confidence_matches),
                'diagnostic_relevance_score': relevance_score,
                'relevance_classification': relevance_class,
                'routing_decision': routing_decision,
                'confidence_threshold': confidence_threshold,
                'user_input_processed': self._preprocess_text(user_input)
            }
        }
    
    def _generate_response_message(self, matches, routing_decision, max_confidence):
        """Generate appropriate response message based on results"""
        if not matches:
            return "No diagnostic codes found - consulting general automotive knowledge"
        
        count = len(matches)
        confidence_note = f"(highest confidence: {max_confidence:.1f}%)"
        
        if routing_decision == 'NLP_TO_LLM':
            if count == 1:
                return f"Found 1 high-confidence diagnostic code {confidence_note} - analyzing with AI"
            else:
                return f"Found {count} potential diagnostic codes {confidence_note} - analyzing with AI"
        else:
            if count == 1:
                return f"Found 1 low-confidence diagnostic code {confidence_note} - providing general guidance"
            else:
                return f"Found {count} low-confidence diagnostic codes {confidence_note} - providing general guidance"
    
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
    obd_codes_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'obd_codes.json')
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