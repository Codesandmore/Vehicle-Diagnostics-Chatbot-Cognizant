"""
YouTube Video Search Utilities
Helper functions for query building, video formatting, and response processing
"""

import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def build_search_query(obd_codes: List[str] = None, symptoms: List[str] = None, user_prompt: str = None) -> List[str]:
    """
    Build optimized search queries for YouTube based on diagnostic information
    
    Args:
        obd_codes: List of OBD diagnostic codes
        symptoms: List of detected symptoms
        user_prompt: Original user prompt
        
    Returns:
        List of search query strings ordered by priority
    """
    queries = []
    
    # Priority 1: Specific OBD code searches
    if obd_codes:
        for code in obd_codes[:3]:  # Limit to top 3 codes
            queries.extend([
                f"{code} repair tutorial",
                f"{code} fix diagnosis",
                f"how to fix {code} error code",
                f"{code} troubleshooting guide"
            ])
    
    # Priority 2: Symptom-based searches
    if symptoms:
        for symptom in symptoms[:2]:  # Limit to top 2 symptoms
            clean_symptom = clean_text_for_search(symptom)
            queries.extend([
                f"car {clean_symptom} repair",
                f"vehicle {clean_symptom} troubleshooting",
                f"automotive {clean_symptom} fix"
            ])
    
    # Priority 3: User prompt-based search
    if user_prompt:
        clean_prompt = extract_keywords_from_prompt(user_prompt)
        if clean_prompt:
            queries.extend([
                f"{clean_prompt} car repair",
                f"{clean_prompt} automotive troubleshooting",
                f"how to fix {clean_prompt}"
            ])
    
    # Priority 4: General fallback searches
    if not queries:
        queries = [
            "car troubleshooting basics",
            "automotive diagnostic guide",
            "car repair tutorials"
        ]
    
    # Remove duplicates while preserving order
    unique_queries = []
    seen = set()
    for query in queries:
        if query.lower() not in seen:
            unique_queries.append(query)
            seen.add(query.lower())
    
    return unique_queries[:10]  # Limit to top 10 queries

def clean_text_for_search(text: str) -> str:
    """
    Clean and optimize text for YouTube search
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text suitable for search
    """
    if not text:
        return ""
    
    # Remove special characters and normalize
    cleaned = re.sub(r'[^\w\s-]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Remove common stopwords that don't help with automotive searches
    stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to'}
    words = [word for word in cleaned.lower().split() if word not in stopwords]
    
    return ' '.join(words)

def extract_keywords_from_prompt(prompt: str) -> str:
    """
    Extract automotive keywords from user prompt
    
    Args:
        prompt: User's original prompt
        
    Returns:
        Extracted keywords for search
    """
    if not prompt:
        return ""
    
    # Common automotive terms to prioritize
    automotive_terms = {
        'engine', 'motor', 'transmission', 'brake', 'brakes', 'suspension', 
        'steering', 'exhaust', 'muffler', 'alternator', 'battery', 'starter',
        'radiator', 'coolant', 'oil', 'filter', 'spark', 'plug', 'ignition',
        'fuel', 'pump', 'injector', 'sensor', 'valve', 'belt', 'hose',
        'tire', 'wheel', 'clutch', 'gear', 'differential', 'axle'
    }
    
    # Extract automotive-related words
    words = clean_text_for_search(prompt).lower().split()
    automotive_words = [word for word in words if word in automotive_terms]
    other_words = [word for word in words if word not in automotive_terms and len(word) > 2]
    
    # Prioritize automotive terms
    keywords = automotive_words + other_words[:3]  # Max 3 additional words
    
    return ' '.join(keywords[:5])  # Limit to 5 total keywords

def format_video_response(videos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format video search results for frontend display
    
    Args:
        videos: List of video dictionaries from YouTube API
        
    Returns:
        Formatted response with rich card data
    """
    if not videos:
        return {
            'has_videos': False,
            'count': 0,
            'videos': [],
            'message': 'No relevant repair videos found for this issue.'
        }
    
    formatted_videos = []
    
    for video in videos:
        try:
            video_data = {
                'id': video.get('id', ''),
                'title': clean_video_title(video.get('title', 'Untitled Video')),
                'description': truncate_description(video.get('description', '')),
                'url': f"https://www.youtube.com/watch?v={video.get('id', '')}",
                'thumbnail_url': get_best_thumbnail(video.get('thumbnails', {})),
                'channel_title': video.get('channel_name', 'Unknown Channel'),
                'duration': format_duration(video.get('duration', '')),
                'view_count': format_view_count(video.get('view_count', 0)),
                'published_at': format_publish_date(video.get('published_at', '')),
                'is_trusted': video.get('is_trusted_channel', False)
            }
            formatted_videos.append(video_data)
            
        except Exception as e:
            logger.warning(f"Error formatting video: {e}")
            continue
    
    return {
        'has_videos': len(formatted_videos) > 0,
        'count': len(formatted_videos),
        'videos': formatted_videos,
        'message': f"Found {len(formatted_videos)} helpful repair video{'s' if len(formatted_videos) != 1 else ''}:"
    }

def clean_video_title(title: str) -> str:
    """Clean video title for display"""
    if not title:
        return "Untitled Video"
    
    # Remove excessive caps and clean up
    if title.isupper():
        title = title.title()
    
    # Truncate if too long
    if len(title) > 80:
        title = title[:77] + "..."
    
    return title

def truncate_description(description: str, max_length: int = 150) -> str:
    """Truncate video description for display"""
    if not description:
        return "No description available."
    
    if len(description) <= max_length:
        return description
    
    # Find last complete sentence or word
    truncated = description[:max_length]
    last_period = truncated.rfind('.')
    last_space = truncated.rfind(' ')
    
    if last_period > max_length - 50:
        return description[:last_period + 1]
    elif last_space > max_length - 20:
        return description[:last_space] + "..."
    else:
        return description[:max_length - 3] + "..."

def get_best_thumbnail(thumbnails: Dict[str, Any]) -> str:
    """Get the best available thumbnail URL"""
    if not thumbnails:
        # Fallback to a default automotive-themed placeholder
        return "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='320' height='180' viewBox='0 0 320 180'%3E%3Crect width='320' height='180' fill='%23f8f9fa'/%3E%3Ctext x='160' y='90' text-anchor='middle' fill='%23666' font-family='Arial' font-size='14'%3E🎥 Video Thumbnail%3C/text%3E%3C/svg%3E"
    
    # Prefer medium, then high, then default
    for quality in ['medium', 'high', 'default']:
        if quality in thumbnails and 'url' in thumbnails[quality]:
            return thumbnails[quality]['url']
    
    # Fallback if no standard thumbnails found
    return "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='320' height='180' viewBox='0 0 320 180'%3E%3Crect width='320' height='180' fill='%23f8f9fa'/%3E%3Ctext x='160' y='90' text-anchor='middle' fill='%23666' font-family='Arial' font-size='14'%3E🎥 Video Thumbnail%3C/text%3E%3C/svg%3E"

def format_duration(duration: str) -> str:
    """Format video duration for display"""
    if not duration:
        return "Unknown"
    
    # Parse ISO 8601 duration (PT4M13S -> 4:13)
    try:
        import re
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
        if match:
            hours, minutes, seconds = match.groups()
            hours = int(hours) if hours else 0
            minutes = int(minutes) if minutes else 0
            seconds = int(seconds) if seconds else 0
            
            if hours > 0:
                return f"{hours}:{minutes:02d}:{seconds:02d}"
            else:
                return f"{minutes}:{seconds:02d}"
    except:
        pass
    
    return duration

def format_view_count(view_count: int) -> str:
    """Format view count for display"""
    if not view_count:
        return "0 views"
    
    if view_count >= 1_000_000:
        return f"{view_count / 1_000_000:.1f}M views"
    elif view_count >= 1_000:
        return f"{view_count / 1_000:.1f}K views"
    else:
        return f"{view_count:,} views"

def format_publish_date(published_at: str) -> str:
    """Format publish date for display"""
    if not published_at:
        return "Unknown date"
    
    try:
        # Parse ISO format date
        pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        now = datetime.now(pub_date.tzinfo)
        
        diff = now - pub_date
        
        if diff.days == 0:
            return "Today"
        elif diff.days == 1:
            return "Yesterday"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        elif diff.days < 30:
            weeks = diff.days // 7
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        elif diff.days < 365:
            months = diff.days // 30
            return f"{months} month{'s' if months != 1 else ''} ago"
        else:
            years = diff.days // 365
            return f"{years} year{'s' if years != 1 else ''} ago"
            
    except Exception:
        return published_at[:10]  # Return just the date part

def is_video_relevant(video_data: Dict[str, Any], search_query: str) -> bool:
    """
    Check if video is relevant to the search query
    
    Args:
        video_data: Video information from YouTube API
        search_query: Original search query
        
    Returns:
        True if video appears relevant, False otherwise
    """
    title = video_data.get('title', '').lower()
    description = video_data.get('description', '').lower()
    query_words = search_query.lower().split()
    
    # Check if at least 50% of query words appear in title or description
    matches = 0
    for word in query_words:
        if word in title or word in description:
            matches += 1
    
    relevance_score = matches / len(query_words) if query_words else 0
    return relevance_score >= 0.3  # At least 30% word match
