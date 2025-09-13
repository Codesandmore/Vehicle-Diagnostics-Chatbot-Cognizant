"""
Speech Recognition Utilities
Handles audio file validation, format conversion, and cleanup
"""

import os
import tempfile
from typing import Optional, Tuple
import logging

# Configure logging for speech module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioValidator:
    """Validates and processes audio files for speech recognition"""
    
    # Supported audio formats
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.webm', '.ogg', '.flac'}
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB max file size
    MAX_DURATION = 300  # 5 minutes max duration
    
    @staticmethod
    def validate_audio_file(file_path: str) -> Tuple[bool, str]:
        """
        Validate audio file format, size, and basic properties
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return False, "Audio file not found"
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > AudioValidator.MAX_FILE_SIZE:
                return False, f"Audio file too large. Max size: {AudioValidator.MAX_FILE_SIZE // (1024*1024)}MB"
            
            if file_size == 0:
                return False, "Audio file is empty"
            
            # Check file extension
            _, ext = os.path.splitext(file_path.lower())
            if ext not in AudioValidator.SUPPORTED_FORMATS:
                return False, f"Unsupported audio format. Supported: {', '.join(AudioValidator.SUPPORTED_FORMATS)}"
            
            logger.info(f"Audio file validated: {file_path} ({file_size} bytes)")
            return True, "Valid audio file"
            
        except Exception as e:
            logger.error(f"Error validating audio file: {e}")
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def validate_file(file_storage) -> Tuple[bool, str]:
        """
        Validate a Flask FileStorage object (uploaded file)
        
        Args:
            file_storage: Flask FileStorage object from request.files
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if file has a filename
            if not file_storage.filename:
                return False, "No filename provided"
            
            # Check file extension
            _, ext = os.path.splitext(file_storage.filename.lower())
            if ext not in AudioValidator.SUPPORTED_FORMATS:
                return False, f"Unsupported audio format. Supported: {', '.join(AudioValidator.SUPPORTED_FORMATS)}"
            
            # Try to get file size if possible
            # Note: For FileStorage, we'll validate size during save
            logger.info(f"File upload validated: {file_storage.filename}")
            return True, "Valid audio file upload"
            
        except Exception as e:
            logger.error(f"Error validating file upload: {e}")
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def create_temp_audio_file(audio_data: bytes, original_filename: str = "audio.webm") -> Optional[str]:
        """
        Create a temporary audio file from uploaded data
        
        Args:
            audio_data: Raw audio bytes
            original_filename: Original filename for extension detection
            
        Returns:
            Path to temporary file or None if failed
        """
        try:
            # Get file extension
            _, ext = os.path.splitext(original_filename.lower())
            if not ext or ext not in AudioValidator.SUPPORTED_FORMATS:
                ext = '.webm'  # Default for browser recordings
            
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix=ext, prefix='speech_')
            
            # Write audio data
            with os.fdopen(temp_fd, 'wb') as temp_file:
                temp_file.write(audio_data)
            
            logger.info(f"Created temporary audio file: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating temporary audio file: {e}")
            return None
    
    @staticmethod
    def cleanup_temp_file(file_path: str) -> bool:
        """
        Safely remove temporary audio file
        
        Args:
            file_path: Path to temporary file
            
        Returns:
            True if cleanup successful
        """
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cleaning up temporary file {file_path}: {e}")
            return False

class AudioProcessor:
    """Handles audio preprocessing and format conversion"""
    
    @staticmethod
    def save_temp_file(file_storage) -> str:
        """
        Save Flask FileStorage to a temporary file
        
        Args:
            file_storage: Flask FileStorage object from request.files
            
        Returns:
            Path to saved temporary file
        """
        try:
            # Get file extension
            _, ext = os.path.splitext(file_storage.filename.lower())
            if not ext:
                ext = '.webm'  # Default for browser recordings
            
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix=ext, prefix='audio_')
            
            # Save uploaded file to temporary location
            with os.fdopen(temp_fd, 'wb') as tmp_file:
                file_storage.save(tmp_file)
            
            # Validate file size after saving
            file_size = os.path.getsize(temp_path)
            if file_size > AudioValidator.MAX_FILE_SIZE:
                os.unlink(temp_path)
                raise ValueError(f"File too large: {file_size} bytes > {AudioValidator.MAX_FILE_SIZE} bytes")
            
            if file_size == 0:
                os.unlink(temp_path)
                raise ValueError("Empty file uploaded")
            
            logger.info(f"Saved temporary audio file: {temp_path} ({file_size} bytes)")
            
            # For WebM files, try to create a more compatible version
            if ext == '.webm':
                try:
                    # Verify the file is readable
                    with open(temp_path, 'rb') as f:
                        header = f.read(10)
                        if len(header) < 4:
                            raise ValueError("Invalid WebM file: too short")
                    logger.info("WebM file validation passed")
                except Exception as webm_error:
                    logger.warning(f"WebM file validation issue: {webm_error}")
                    # Continue anyway, let Whisper try to handle it
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Error saving temporary file: {e}")
            raise
    
    @staticmethod
    def cleanup_temp_file(temp_path: str) -> bool:
        """
        Clean up temporary audio file
        
        Args:
            temp_path: Path to temporary file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.info(f"Cleaned up temporary file: {temp_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cleaning up temporary file {temp_path}: {e}")
            return False
    
    @staticmethod
    def convert_to_wav(input_path: str) -> Optional[str]:
        """
        Convert audio file to WAV format for better Whisper compatibility
        
        Args:
            input_path: Path to input audio file
            
        Returns:
            Path to converted WAV file or None if failed
        """
        try:
            # Check if it's already a WAV file
            if input_path.lower().endswith('.wav'):
                return input_path
            
            # Try to convert using pydub
            try:
                from pydub import AudioSegment
                
                # Load the audio file
                logger.info(f"Converting {input_path} to WAV format...")
                audio = AudioSegment.from_file(input_path)
                
                # Create a new temporary WAV file
                wav_fd, wav_path = tempfile.mkstemp(suffix='.wav', prefix='audio_wav_')
                os.close(wav_fd)  # Close the file descriptor
                
                # Export as WAV
                audio.export(wav_path, format="wav")
                
                # Verify the WAV file was created
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    logger.info(f"Successfully converted to WAV: {wav_path}")
                    return wav_path
                else:
                    logger.error("WAV conversion failed - file not created or empty")
                    return None
                    
            except ImportError:
                logger.warning("pydub not available for audio conversion")
                return input_path  # Return original if no conversion possible
            except Exception as conv_error:
                logger.error(f"Audio conversion error: {conv_error}")
                return input_path  # Return original if conversion fails
                
        except Exception as e:
            logger.error(f"Error in convert_to_wav: {e}")
            return input_path  # Return original path as fallback
    
    @staticmethod
    def get_audio_info(file_path: str) -> dict:
        """
        Get basic audio file information
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            file_size = os.path.getsize(file_path)
            _, ext = os.path.splitext(file_path.lower())
            
            return {
                'file_path': file_path,
                'file_size': file_size,
                'format': ext,
                'size_mb': round(file_size / (1024 * 1024), 2)
            }
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return {}

def sanitize_transcript(text: str) -> str:
    """
    Clean and sanitize the transcribed text
    
    Args:
        text: Raw transcript from Whisper
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Basic text cleaning
    cleaned = text.strip()
    
    # Remove excessive whitespace
    import re
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Capitalize first letter
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    return cleaned
