"""
OpenAI Whisper Transcription Service

Uses OpenAI's Whisper API to transcribe audio files.
Supports long recordings by chunking audio.
"""

import os
import io
import logging
import tempfile
import time
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Retry config for Whisper API calls
_RETRY_STATUS_CODES = {429, 502, 503}
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds — doubles each attempt (1s, 2s, 4s)


class WhisperService:
    """OpenAI Whisper transcription service."""

    TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"
    DEFAULT_MODEL = "whisper-1"
    # Whisper API has a 25MB file size limit
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        """Initialize the Whisper service.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use for transcription
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")

        self.model = model
        logger.info(f"Whisper service initialized with model: {model}")

    def transcribe(
        self,
        audio_data: bytes,
        filename: str = "audio.webm",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Transcribe audio data using Whisper API.

        Args:
            audio_data: Raw audio bytes
            filename: Original filename (helps API detect format)
            language: Optional language code (e.g., 'en')
            prompt: Optional prompt to guide transcription

        Returns:
            Tuple of (success: bool, transcript_or_error: str)
        """
        if not audio_data:
            return False, "No audio data provided"

        if len(audio_data) > self.MAX_FILE_SIZE:
            return False, f"Audio file too large (max {self.MAX_FILE_SIZE // (1024*1024)}MB)"

        # Determine file extension for Content-Type
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'webm'

        # Map extension to MIME type
        mime_types = {
            'webm': 'audio/webm',
            'mp3': 'audio/mpeg',
            'mp4': 'audio/mp4',
            'm4a': 'audio/mp4',
            'wav': 'audio/wav',
            'ogg': 'audio/ogg',
            'flac': 'audio/flac',
        }
        mime_type = mime_types.get(ext, 'audio/webm')

        # Prepare multipart form data (built once, reused across retries)
        data = {
            'model': self.model,
            'response_format': 'text',  # Plain text response
        }
        if language:
            data['language'] = language
        if prompt:
            data['prompt'] = prompt

        logger.info(f"Sending audio to Whisper API: {len(audio_data)} bytes, format={ext}")

        last_error_msg = "Unknown error"
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                # Rebuild files dict each attempt — requests consumes the bytes object
                files = {
                    'file': (filename, audio_data, mime_type)
                }

                response = requests.post(
                    self.TRANSCRIPTION_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    files=files,
                    data=data,
                    timeout=300,  # 5 minute timeout for long recordings
                )

                if response.status_code == 200:
                    transcript = response.text.strip()
                    logger.info(f"Transcription successful: {len(transcript)} chars (attempt {attempt})")
                    return True, transcript

                if response.status_code in _RETRY_STATUS_CODES and attempt < _MAX_RETRIES:
                    # Respect Retry-After header on 429 rate limit responses
                    retry_after = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    if response.status_code == 429:
                        header_val = response.headers.get("Retry-After")
                        if header_val:
                            try:
                                retry_after = float(header_val)
                            except ValueError:
                                pass
                    logger.warning(
                        f"Whisper API returned {response.status_code} on attempt {attempt}/{_MAX_RETRIES}. "
                        f"Retrying in {retry_after:.1f}s..."
                    )
                    time.sleep(retry_after)
                    continue

                # Non-retryable error or final attempt
                last_error_msg = f"Whisper API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        last_error_msg = error_data['error'].get('message', last_error_msg)
                except Exception:
                    last_error_msg = response.text[:200] if response.text else last_error_msg
                logger.error(f"Whisper API failed (attempt {attempt}/{_MAX_RETRIES}): {last_error_msg}")
                if attempt == _MAX_RETRIES or response.status_code not in _RETRY_STATUS_CODES:
                    return False, last_error_msg

            except requests.exceptions.Timeout:
                last_error_msg = "Transcription timed out. Try a shorter recording."
                logger.error(f"Whisper API timed out (attempt {attempt}/{_MAX_RETRIES})")
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            except requests.exceptions.RequestException as e:
                last_error_msg = f"Network error: {str(e)}"
                logger.error(f"Whisper API request failed (attempt {attempt}/{_MAX_RETRIES}): {e}")
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            except Exception as e:
                logger.error(f"Unexpected error during transcription (attempt {attempt}/{_MAX_RETRIES}): {e}")
                return False, f"Transcription error: {str(e)}"

        logger.error(f"Whisper API failed after {_MAX_RETRIES} attempts: {last_error_msg}")
        return False, last_error_msg

    def is_available(self) -> bool:
        """Check if the Whisper service is available."""
        return bool(self.api_key)


# Singleton instance (lazy initialization)
_whisper_service: Optional[WhisperService] = None


def get_whisper_service() -> Optional[WhisperService]:
    """Get or create the Whisper service singleton.

    Returns:
        WhisperService instance or None if API key not configured
    """
    global _whisper_service
    if _whisper_service is None:
        try:
            _whisper_service = WhisperService()
        except ValueError:
            logger.warning("Whisper service not available: OPENAI_API_KEY not set")
            return None
    return _whisper_service
