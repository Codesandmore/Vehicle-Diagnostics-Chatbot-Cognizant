"""
YouTube Video Search Module
Provides auto-suggestion and manual search capabilities for vehicle diagnostic videos
"""

from .youtube_handler import YouTubeVideoSearcher, search_diagnostic_videos
from .youtube_utils import format_video_response, build_search_query
from .youtube_config import get_youtube_service, is_youtube_available

__all__ = [
    'YouTubeVideoSearcher',
    'search_diagnostic_videos', 
    'format_video_response',
    'build_search_query',
    'get_youtube_service',
    'is_youtube_available'
]
