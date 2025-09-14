"""
YouTube API Configuration and Service Setup
Handles API key management and service initialization
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class YouTubeConfig:
    """Configuration management for YouTube API"""
    
    # Trusted automotive channels for credibility filtering
    TRUSTED_CHANNELS = [
        'ChrisFix',
        'EricTheCarGuy', 
        'Scotty Kilmer',
        'Car Care Nut',
        'ScannerDanner',
        'South Main Auto Repair LLC',
        'Pine Hollow Auto Diagnostics',
        'Automotive Service Training',
        'BMW TechnicianMD',
        'Mercedes Medic',
        'FordTechMakuloco',
        'Honest John',
        'RepairPal',
        'AutoZone',
        'Advance Auto Parts'
    ]
    
    # Search preferences
    MAX_RESULTS = 5
    PREFERRED_DURATION = 'medium'  # short (< 4 min), medium (4-20 min), long (> 20 min)
    MIN_VIEW_COUNT = 1000
    
    @staticmethod
    def get_api_key() -> Optional[str]:
        """Get YouTube API key from environment variables"""
        return os.getenv('YOUTUBE_API_KEY')
    
    @staticmethod
    def is_configured() -> bool:
        """Check if YouTube API is properly configured"""
        return bool(YouTubeConfig.get_api_key())

def get_youtube_service():
    """
    Initialize YouTube API service
    
    Returns:
        YouTube service object or None if not available
    """
    try:
        from googleapiclient.discovery import build
        
        api_key = YouTubeConfig.get_api_key()
        if not api_key:
            logger.warning("YouTube API key not found in environment variables")
            return None
        
        youtube = build('youtube', 'v3', developerKey=api_key)
        logger.info("✅ YouTube API service initialized successfully")
        return youtube
        
    except ImportError:
        logger.error("google-api-python-client not installed. Run: pip install google-api-python-client")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize YouTube API service: {e}")
        return None

def is_youtube_available() -> bool:
    """
    Check if YouTube functionality is available
    
    Returns:
        True if YouTube can be used, False otherwise
    """
    try:
        service = get_youtube_service()
        return service is not None
    except Exception:
        return False

def validate_youtube_setup() -> dict:
    """
    Validate YouTube API setup and return status
    
    Returns:
        Dictionary with setup status information
    """
    status = {
        'available': False,
        'api_key_configured': False,
        'dependencies_installed': False,
        'service_initialized': False,
        'error': None
    }
    
    try:
        # Check API key
        status['api_key_configured'] = bool(YouTubeConfig.get_api_key())
        
        # Check dependencies
        try:
            from googleapiclient.discovery import build
            status['dependencies_installed'] = True
        except ImportError:
            status['error'] = "google-api-python-client not installed"
            return status
        
        # Check service initialization
        service = get_youtube_service()
        if service:
            status['service_initialized'] = True
            status['available'] = True
        else:
            status['error'] = "Failed to initialize YouTube service"
            
    except Exception as e:
        status['error'] = str(e)
        
    return status
