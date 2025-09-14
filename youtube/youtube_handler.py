"""
YouTube Video Search Handler
Main interface for searching automotive repair videos with credibility filtering
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from .youtube_config import YouTubeConfig, get_youtube_service
from .youtube_utils import build_search_query, format_video_response, is_video_relevant

logger = logging.getLogger(__name__)

class YouTubeVideoSearcher:
    """Main class for searching automotive repair videos on YouTube"""
    
    def __init__(self):
        self.youtube = get_youtube_service()
        self.config = YouTubeConfig()
        self._trusted_channel_ids = {}  # Cache for channel ID lookups
        
    def is_available(self) -> bool:
        """Check if YouTube search is available"""
        return self.youtube is not None
    
    def search_diagnostic_videos(self, 
                                obd_codes: List[str] = None, 
                                symptoms: List[str] = None, 
                                user_prompt: str = None,
                                max_results: int = None) -> Dict[str, Any]:
        """
        Search for diagnostic repair videos based on OBD codes and symptoms
        
        Args:
            obd_codes: List of OBD diagnostic codes
            symptoms: List of detected symptoms
            user_prompt: Original user prompt
            max_results: Maximum number of videos to return
            
        Returns:
            Formatted response with video results
        """
        if not self.is_available():
            logger.warning("YouTube service not available")
            return format_video_response([])
        
        max_results = max_results or self.config.MAX_RESULTS
        
        try:
            # Build search queries
            search_queries = build_search_query(obd_codes, symptoms, user_prompt)
            
            if not search_queries:
                logger.info("No search queries generated")
                return format_video_response([])
            
            # Search videos using multiple queries
            all_videos = []
            seen_video_ids = set()
            
            for query in search_queries[:3]:  # Try top 3 queries
                videos = self._search_videos(query, max_results=3)
                
                for video in videos:
                    video_id = video.get('id')
                    if video_id and video_id not in seen_video_ids:
                        seen_video_ids.add(video_id)
                        all_videos.append(video)
                
                if len(all_videos) >= max_results:
                    break
            
            # Filter by credibility and relevance
            filtered_videos = self._filter_videos(all_videos, search_queries[0] if search_queries else "")
            
            # Limit results
            final_videos = filtered_videos[:max_results]
            
            logger.info(f"Found {len(final_videos)} videos for diagnostic search")
            return format_video_response(final_videos)
            
        except Exception as e:
            logger.error(f"Error searching diagnostic videos: {e}")
            return format_video_response([])
    
    def search_manual_videos(self, query: str, max_results: int = None) -> Dict[str, Any]:
        """
        Manual search for automotive videos
        
        Args:
            query: Search query string
            max_results: Maximum number of videos to return
            
        Returns:
            Formatted response with video results
        """
        if not self.is_available():
            logger.warning("YouTube service not available")
            return format_video_response([])
        
        max_results = max_results or self.config.MAX_RESULTS
        
        try:
            # Enhance query with automotive context
            enhanced_query = f"{query} car automotive repair"
            
            videos = self._search_videos(enhanced_query, max_results=max_results * 2)
            filtered_videos = self._filter_videos(videos, query)
            final_videos = filtered_videos[:max_results]
            
            logger.info(f"Found {len(final_videos)} videos for manual search: {query}")
            return format_video_response(final_videos)
            
        except Exception as e:
            logger.error(f"Error in manual video search: {e}")
            return format_video_response([])
    
    def _search_videos(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform YouTube API search
        
        Args:
            query: Search query
            max_results: Maximum results to fetch
            
        Returns:
            List of video data dictionaries
        """
        try:
            # Search for videos
            search_response = self.youtube.search().list(
                q=query,
                part='snippet',
                type='video',
                maxResults=min(max_results, 25),  # API limit
                order='relevance',
                videoDuration=self.config.PREFERRED_DURATION,
                videoDefinition='any',
                videoEmbeddable='true'
            ).execute()
            
            videos = []
            video_ids = []
            
            # Extract basic info
            for search_result in search_response.get('items', []):
                video_id = search_result['id']['videoId']
                video_ids.append(video_id)
                
                snippet = search_result['snippet']
                video_data = {
                    'id': video_id,
                    'title': snippet.get('title', ''),
                    'description': snippet.get('description', ''),
                    'channel_name': snippet.get('channelTitle', ''),
                    'channel_id': snippet.get('channelId', ''),
                    'published_at': snippet.get('publishedAt', ''),
                    'thumbnails': snippet.get('thumbnails', {}),
                }
                videos.append(video_data)
            
            # Get additional video details in batch
            if video_ids:
                self._enrich_video_data(videos, video_ids)
            
            return videos
            
        except Exception as e:
            logger.error(f"YouTube API search error: {e}")
            return []
    
    def _enrich_video_data(self, videos: List[Dict[str, Any]], video_ids: List[str]):
        """
        Enrich video data with additional details from YouTube API
        
        Args:
            videos: List of video dictionaries to enrich
            video_ids: List of video IDs to fetch details for
        """
        try:
            # Get video statistics and content details
            videos_response = self.youtube.videos().list(
                part='statistics,contentDetails',
                id=','.join(video_ids)
            ).execute()
            
            # Create lookup dict
            video_details = {
                item['id']: item for item in videos_response.get('items', [])
            }
            
            # Enrich video data
            for video in videos:
                video_id = video.get('id')
                if video_id in video_details:
                    details = video_details[video_id]
                    
                    # Add statistics
                    stats = details.get('statistics', {})
                    video['view_count'] = int(stats.get('viewCount', 0))
                    video['like_count'] = int(stats.get('likeCount', 0))
                    
                    # Add duration
                    content_details = details.get('contentDetails', {})
                    video['duration'] = content_details.get('duration', '')
                
                # Check if channel is trusted
                video['is_trusted_channel'] = self._is_trusted_channel(
                    video.get('channel_name', ''),
                    video.get('channel_id', '')
                )
                
        except Exception as e:
            logger.warning(f"Error enriching video data: {e}")
    
    def _is_trusted_channel(self, channel_name: str, channel_id: str) -> bool:
        """
        Check if channel is in trusted automotive channels list
        
        Args:
            channel_name: Name of the channel
            channel_id: YouTube channel ID
            
        Returns:
            True if channel is trusted, False otherwise
        """
        if not channel_name:
            return False
        
        # Check against trusted channel names (case insensitive)
        for trusted_name in self.config.TRUSTED_CHANNELS:
            if trusted_name.lower() in channel_name.lower():
                return True
        
        # Cache and check channel IDs if needed
        # (Could implement channel ID lookup for more accurate matching)
        
        return False
    
    def _filter_videos(self, videos: List[Dict[str, Any]], search_query: str) -> List[Dict[str, Any]]:
        """
        Filter videos by credibility, relevance, and quality
        
        Args:
            videos: List of video dictionaries
            search_query: Original search query
            
        Returns:
            Filtered and sorted list of videos
        """
        filtered = []
        
        for video in videos:
            # Basic quality filters
            view_count = video.get('view_count', 0)
            if view_count < self.config.MIN_VIEW_COUNT:
                continue
            
            # Relevance check
            if not is_video_relevant(video, search_query):
                continue
            
            # Channel credibility check (not required but boosts ranking)
            is_trusted = video.get('is_trusted_channel', False)
            video['trust_score'] = 1.0 if is_trusted else 0.5
            
            filtered.append(video)
        
        # Sort by trust score, then view count
        filtered.sort(key=lambda v: (v.get('trust_score', 0), v.get('view_count', 0)), reverse=True)
        
        return filtered

# Convenience function for easy integration
def search_diagnostic_videos(obd_codes: List[str] = None, 
                           symptoms: List[str] = None, 
                           user_prompt: str = None,
                           max_results: int = 5) -> Dict[str, Any]:
    """
    Convenience function to search for diagnostic videos
    
    Args:
        obd_codes: List of OBD diagnostic codes
        symptoms: List of detected symptoms  
        user_prompt: Original user prompt
        max_results: Maximum number of videos to return
        
    Returns:
        Formatted response with video results
    """
    searcher = YouTubeVideoSearcher()
    return searcher.search_diagnostic_videos(
        obd_codes=obd_codes,
        symptoms=symptoms, 
        user_prompt=user_prompt,
        max_results=max_results
    )

def search_manual_videos(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Convenience function to search for videos with manual query
    
    Args:
        query: Search query string
        max_results: Maximum number of videos to return
        
    Returns:
        Formatted response with video results
    """
    searcher = YouTubeVideoSearcher()
    return searcher.search_manual_videos(query=query, max_results=max_results)
