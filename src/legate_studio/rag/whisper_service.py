"""
Gemini Flash Multimodal Transcription Service

Uses Google Gemini Flash to transcribe audio files via multimodal input.
Replaces the previous OpenAI Whisper implementation.

The original WhisperService (OpenAI) is preserved below as a commented-out
fallback. To restore it, uncomment the class and update the singleton to use it.
"""

import logging
import os
import tempfile
import time

logger = logging.getLogger(__name__)

# Retry config
_RETRY_STATUS_CODES = {429, 502, 503}
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds — doubles each attempt (1s, 2s, 4s)

# MIME type map for common audio formats
_MIME_TYPES = {
    "webm": "audio/webm",
    "mp3": "audio/mpeg",
    "mp4": "audio/mp4",
    "m4a": "audio/mp4",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}


class GeminiTranscriptionService:
    """
    Transcription service backed by Gemini Flash multimodal input.

    Public interface is identical to the old WhisperService so call sites
    need only update their import / instantiation.
    """

    # Override with GEMINI_TRANSCRIBE_MODEL so a model retirement can be
    # handled with a config change instead of a code deploy.
    DEFAULT_MODEL = os.environ.get("GEMINI_TRANSCRIBE_MODEL", "gemini-2.5-flash")
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB — keep parity with Whisper limit

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        """
        Args:
            api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
            model:   Gemini model to use for transcription.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided and GEMINI_API_KEY not set")

        self.model = model
        logger.info(f"GeminiTranscriptionService initialized with model: {model}")

    # ------------------------------------------------------------------
    # Public interface (same as WhisperService)
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_data: bytes,
        filename: str = "audio.webm",
        language: str | None = None,
        prompt: str | None = None,
    ) -> tuple[bool, str]:
        """Transcribe audio data using Gemini Flash multimodal input.

        Args:
            audio_data: Raw audio bytes.
            filename:   Original filename — used to infer the MIME type.
            language:   Optional language hint (e.g. 'en'). Appended to prompt.
            prompt:     Optional text hint to guide transcription.

        Returns:
            Tuple of (success: bool, transcript_or_error: str).
        """
        if not audio_data:
            return False, "No audio data provided"

        if len(audio_data) > self.MAX_FILE_SIZE:
            return False, f"Audio file too large (max {self.MAX_FILE_SIZE // (1024 * 1024)}MB)"

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "webm"
        mime_type = _MIME_TYPES.get(ext, "audio/webm")

        # Build the transcription instruction
        instruction_parts = [
            "Transcribe this audio accurately. Return only the transcription text, no commentary."
        ]
        if language:
            instruction_parts.append(f"The audio is in language: {language}.")
        if prompt:
            instruction_parts.append(f"Context hint: {prompt}")
        instruction = " ".join(instruction_parts)

        logger.info(
            f"Sending audio to Gemini for transcription: {len(audio_data)} bytes, "
            f"format={ext}, mime={mime_type}"
        )

        last_error_msg = "Unknown error"

        for attempt in range(1, _MAX_RETRIES + 1):
            tmp_path = None
            uploaded_file = None
            try:
                import google.generativeai as genai

                # Configure with this instance's key (avoids global-state collisions)
                genai.configure(api_key=self.api_key)

                # 1. Write to a temp file so genai.upload_file can read it
                suffix = f".{ext}"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(audio_data)
                    tmp_path = tmp.name

                # 2. Upload via Files API
                uploaded_file = genai.upload_file(tmp_path, mime_type=mime_type)
                logger.debug(f"Uploaded audio to Gemini Files API: {uploaded_file.name}")

                # 3. Generate transcription
                model = genai.GenerativeModel(self.model)
                response = model.generate_content([uploaded_file, instruction])

                transcript = response.text.strip() if response.text else ""
                logger.info(
                    f"Transcription successful: {len(transcript)} chars (attempt {attempt})"
                )
                return True, transcript

            except Exception as e:
                error_str = str(e)
                last_error_msg = error_str

                # Check for retryable HTTP status codes embedded in the exception message
                is_rate_limit = "429" in error_str or "quota" in error_str.lower()
                is_server_error = any(code in error_str for code in ("502", "503", "500"))
                retryable = is_rate_limit or is_server_error

                if retryable and attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"Gemini transcription error on attempt {attempt}/{_MAX_RETRIES} "
                        f"(retryable): {error_str[:200]}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Gemini transcription failed (attempt {attempt}/{_MAX_RETRIES}): "
                        f"{error_str[:300]}"
                    )
                    if not retryable:
                        # Non-retryable — bail immediately
                        return False, f"Transcription error: {error_str}"

            finally:
                # 4. Clean up temp file
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

                # 5. Delete uploaded file from Gemini Files API
                if uploaded_file is not None:
                    try:
                        import google.generativeai as genai  # noqa: F811 — already imported above
                        genai.delete_file(uploaded_file.name)
                        logger.debug(f"Deleted uploaded file: {uploaded_file.name}")
                    except Exception as del_err:
                        logger.warning(f"Could not delete uploaded file {uploaded_file.name}: {del_err}")

        logger.error(f"Gemini transcription failed after {_MAX_RETRIES} attempts: {last_error_msg}")
        return False, last_error_msg

    def is_available(self) -> bool:
        """Check if the transcription service is configured."""
        return bool(self.api_key)


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

# Primary singleton — Gemini-backed
_transcription_service: GeminiTranscriptionService | None = None


def get_transcription_service(api_key: str | None = None) -> GeminiTranscriptionService | None:
    """Get or create the Gemini transcription service singleton.

    Args:
        api_key: Optional API key override (e.g. a BYOK user's Gemini key).
                 When provided a new instance is returned rather than the singleton,
                 so per-user keys are isolated.

    Returns:
        GeminiTranscriptionService instance, or None if no API key is configured.
    """
    global _transcription_service

    # If a specific key is passed, return a fresh (non-cached) instance
    if api_key:
        try:
            return GeminiTranscriptionService(api_key=api_key)
        except ValueError:
            logger.warning("GeminiTranscriptionService: provided api_key was empty")
            return None

    if _transcription_service is None:
        try:
            _transcription_service = GeminiTranscriptionService()
        except ValueError:
            logger.warning(
                "Gemini transcription service not available: GEMINI_API_KEY not set"
            )
            return None

    return _transcription_service


def get_whisper_service(api_key: str | None = None) -> GeminiTranscriptionService | None:
    """Backward-compatibility alias for get_transcription_service().

    Old call sites that imported get_whisper_service() will continue to work
    without modification. New code should prefer get_transcription_service().
    """
    return get_transcription_service(api_key=api_key)


# ---------------------------------------------------------------------------
# Fallback: original OpenAI WhisperService (kept for reference / easy restore)
# ---------------------------------------------------------------------------
#
# class WhisperService:
#     """OpenAI Whisper transcription service (original implementation)."""
#
#     TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"
#     DEFAULT_MODEL = "whisper-1"
#     MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
#
#     def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
#         import requests  # noqa: F401
#         self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
#         if not self.api_key:
#             raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")
#         self.model = model
#         logger.info(f"Whisper service initialized with model: {model}")
#
#     def transcribe(
#         self,
#         audio_data: bytes,
#         filename: str = "audio.webm",
#         language: str | None = None,
#         prompt: str | None = None,
#     ) -> tuple[bool, str]:
#         if not audio_data:
#             return False, "No audio data provided"
#         if len(audio_data) > self.MAX_FILE_SIZE:
#             return False, f"Audio file too large (max {self.MAX_FILE_SIZE // (1024 * 1024)}MB)"
#
#         ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "webm"
#         mime_types = {
#             "webm": "audio/webm", "mp3": "audio/mpeg", "mp4": "audio/mp4",
#             "m4a": "audio/mp4", "wav": "audio/wav", "ogg": "audio/ogg", "flac": "audio/flac",
#         }
#         mime_type = mime_types.get(ext, "audio/webm")
#         data = {"model": self.model, "response_format": "text"}
#         if language:
#             data["language"] = language
#         if prompt:
#             data["prompt"] = prompt
#
#         import requests
#         last_error_msg = "Unknown error"
#         for attempt in range(1, _MAX_RETRIES + 1):
#             try:
#                 files = {"file": (filename, audio_data, mime_type)}
#                 response = requests.post(
#                     self.TRANSCRIPTION_URL,
#                     headers={"Authorization": f"Bearer {self.api_key}"},
#                     files=files, data=data, timeout=300,
#                 )
#                 if response.status_code == 200:
#                     return True, response.text.strip()
#                 if response.status_code in _RETRY_STATUS_CODES and attempt < _MAX_RETRIES:
#                     retry_after = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
#                     if response.status_code == 429:
#                         hdr = response.headers.get("Retry-After")
#                         if hdr:
#                             try:
#                                 retry_after = float(hdr)
#                             except ValueError:
#                                 pass
#                     time.sleep(retry_after)
#                     continue
#                 last_error_msg = f"Whisper API error: {response.status_code}"
#                 try:
#                     err = response.json().get("error", {})
#                     last_error_msg = err.get("message", last_error_msg)
#                 except Exception:
#                     last_error_msg = response.text[:200] or last_error_msg
#                 if attempt == _MAX_RETRIES or response.status_code not in _RETRY_STATUS_CODES:
#                     return False, last_error_msg
#             except requests.exceptions.Timeout:
#                 last_error_msg = "Transcription timed out."
#                 if attempt < _MAX_RETRIES:
#                     time.sleep(_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
#             except requests.exceptions.RequestException as e:
#                 last_error_msg = f"Network error: {e}"
#                 if attempt < _MAX_RETRIES:
#                     time.sleep(_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
#             except Exception as e:
#                 return False, f"Transcription error: {e}"
#         return False, last_error_msg
#
#     def is_available(self) -> bool:
#         return bool(self.api_key)
#
#
# _whisper_service: WhisperService | None = None
#
# def _get_whisper_service_openai() -> WhisperService | None:
#     """Original OpenAI-backed singleton (kept for fallback reference)."""
#     global _whisper_service
#     if _whisper_service is None:
#         try:
#             _whisper_service = WhisperService()
#         except ValueError:
#             logger.warning("Whisper service not available: OPENAI_API_KEY not set")
#             return None
#     return _whisper_service
