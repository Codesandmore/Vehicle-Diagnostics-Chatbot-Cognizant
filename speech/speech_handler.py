"""
Speech Recognition Handler using OpenAI Whisper
Handles server-side speech-to-text conversion
"""

import os
import logging
from typing import Optional, Dict, Any
from .speech_utils import AudioValidator, AudioProcessor, sanitize_transcript

# Configure logging
logger = logging.getLogger(__name__)

class WhisperSpeechHandler:
    """Main speech recognition handler using OpenAI Whisper"""
    
    def __init__(self):
        self.whisper_available = False
        self.model = None
        self.model_name = "base"  # Start with base model for faster processing
        
        # Try to initialize Whisper
        self._initialize_whisper()
    
    def _initialize_whisper(self):
        """Initialize OpenAI Whisper model"""
        try:
            import whisper
            
            # Load the model
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            self.whisper_available = True
            
            logger.info("✅ Whisper speech recognition initialized successfully")
            
        except ImportError:
            logger.warning("❌ OpenAI Whisper not found. Please install: pip install openai-whisper")
            self.whisper_available = False
        except Exception as e:
            logger.error(f"❌ Error initializing Whisper: {e}")
            self.whisper_available = False
    
    def is_available(self) -> bool:
        """Check if Whisper is available for use"""
        return self.whisper_available and self.model is not None
    
    def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file to text using Whisper
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary with transcription results and metadata
        """
        if not self.is_available():
            return {
                'success': False,
                'error': 'Whisper speech recognition is not available',
                'text': '',
                'confidence': 0.0
            }
        
        try:
            # Validate audio file
            is_valid, validation_message = AudioValidator.validate_audio_file(audio_file_path)
            if not is_valid:
                return {
                    'success': False,
                    'error': f'Audio validation failed: {validation_message}',
                    'text': '',
                    'confidence': 0.0
                }
            
            # Get audio info for logging
            audio_info = AudioProcessor.get_audio_info(audio_file_path)
            logger.info(f"Transcribing audio: {audio_info}")
            
            # Try to convert to WAV for better compatibility
            wav_path = None
            original_path = audio_file_path
            
            try:
                wav_path = AudioProcessor.convert_to_wav(audio_file_path)
                if wav_path and wav_path != audio_file_path:
                    logger.info(f"Using converted WAV file: {wav_path}")
                    audio_file_path = wav_path
                else:
                    logger.info(f"Using original file format: {audio_file_path}")
            except Exception as conv_error:
                logger.warning(f"Audio conversion failed, using original: {conv_error}")
                audio_file_path = original_path
            
            # Check if file actually exists before transcription
            if not os.path.exists(audio_file_path):
                return {
                    'success': False,
                    'error': f'Audio file not found: {audio_file_path}',
                    'text': '',
                    'confidence': 0.0
                }
            
            # Additional debugging - check file accessibility
            try:
                with open(audio_file_path, 'rb') as f:
                    file_size = len(f.read())
                logger.info(f"File accessibility check passed: {file_size} bytes")
            except Exception as file_error:
                logger.error(f"File accessibility issue: {file_error}")
                return {
                    'success': False,
                    'error': f'File access error: {file_error}',
                    'text': '',
                    'confidence': 0.0
                }
            
            # Transcribe with Whisper
            logger.info("🎤 Starting Whisper transcription...")
            logger.info(f"Whisper model type: {type(self.model)}")
            logger.info(f"File path: {audio_file_path}")
            
            try:
                # Try the simplest possible transcription first
                logger.info("Attempting basic transcription...")
                result = self.model.transcribe(audio_file_path, fp16=False)
                
            except Exception as transcribe_error:
                logger.error(f"❌ Whisper transcription error: {transcribe_error}")
                logger.error(f"Error type: {type(transcribe_error)}")
                
                # Try with explicit format specification
                try:
                    logger.info("🔄 Retrying with format specification...")
                    import whisper
                    # Convert file path to string in case it's a Path object
                    str_path = str(audio_file_path)
                    result = self.model.transcribe(str_path, fp16=False, word_timestamps=False)
                    
                except Exception as retry_error:
                    logger.error(f"❌ Retry transcription failed: {retry_error}")
                    logger.error(f"Retry error type: {type(retry_error)}")
                    
                    # Final fallback - try to diagnose the issue
                    try:
                        logger.info("🔍 Diagnosing Whisper installation...")
                        import whisper
                        logger.info(f"Whisper version: {whisper.__version__ if hasattr(whisper, '__version__') else 'unknown'}")
                        
                        # Check if this is an ffmpeg issue
                        if "ffmpeg" in str(retry_error).lower() or "system cannot find" in str(retry_error).lower():
                            return {
                                'success': False,
                                'error': 'Audio transcription failed: FFmpeg may not be installed or accessible. WebM files require FFmpeg for processing.',
                                'text': '',
                                'confidence': 0.0
                            }
                        
                    except Exception as diag_error:
                        logger.error(f"Diagnostic error: {diag_error}")
                    
                    return {
                        'success': False,
                        'error': f'Transcription failed: {retry_error}',
                        'text': '',
                        'confidence': 0.0
                    }
            
            # Extract and clean the text
            raw_text = result.get('text', '').strip()
            cleaned_text = sanitize_transcript(raw_text)
            
            # Calculate confidence (Whisper doesn't provide direct confidence, so we estimate)
            confidence = self._estimate_confidence(result)
            
            logger.info(f"✅ Transcription completed: '{cleaned_text}' (confidence: {confidence:.2f})")
            
            # Clean up converted WAV file if it was created
            if wav_path and wav_path != original_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                    logger.info(f"Cleaned up converted WAV file: {wav_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup WAV file: {cleanup_error}")
            
            return {
                'success': True,
                'text': cleaned_text,
                'raw_text': raw_text,
                'confidence': confidence,
                'language': result.get('language', 'en'),
                'audio_info': audio_info
            }
            
        except Exception as e:
            logger.error(f"❌ Error during Whisper transcription: {e}")
            
            # Clean up converted WAV file if it was created
            if wav_path and wav_path != original_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                    logger.info(f"Cleaned up converted WAV file after error: {wav_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup WAV file after error: {cleanup_error}")
            
            return {
                'success': False,
                'error': f'Transcription failed: {str(e)}',
                'text': '',
                'confidence': 0.0
            }
    
    def _estimate_confidence(self, whisper_result: dict) -> float:
        """
        Estimate confidence score from Whisper result
        
        Args:
            whisper_result: Result dictionary from Whisper
            
        Returns:
            Estimated confidence score (0.0 to 1.0)
        """
        try:
            # Whisper doesn't provide direct confidence scores
            # We estimate based on text length and presence of segments
            text = whisper_result.get('text', '').strip()
            segments = whisper_result.get('segments', [])
            
            if not text:
                return 0.0
            
            # Base confidence on text characteristics
            confidence = 0.5  # Base confidence
            
            # Boost confidence for longer, coherent text
            if len(text) > 10:
                confidence += 0.2
            
            if len(text) > 30:
                confidence += 0.1
            
            # Boost confidence if we have segment data
            if segments:
                confidence += 0.1
            
            # Reduce confidence for very short text
            if len(text) < 5:
                confidence -= 0.2
            
            # Ensure confidence is between 0 and 1
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.5  # Default confidence
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the speech recognition system
        
        Returns:
            Status dictionary
        """
        return {
            'whisper_available': self.whisper_available,
            'model_loaded': self.model is not None,
            'model_name': self.model_name,
            'supported_formats': list(AudioValidator.SUPPORTED_FORMATS),
            'max_file_size_mb': AudioValidator.MAX_FILE_SIZE // (1024 * 1024),
            'max_duration_seconds': AudioValidator.MAX_DURATION
        }

# Global speech handler instance
speech_handler = WhisperSpeechHandler()

def transcribe_audio_file(audio_file_path: str) -> Dict[str, Any]:
    """
    Convenience function to transcribe an audio file
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        Transcription result dictionary
    """
    return speech_handler.transcribe_audio(audio_file_path)

def get_speech_status() -> Dict[str, Any]:
    """
    Get speech recognition system status
    
    Returns:
        Status dictionary
    """
    return speech_handler.get_status()
