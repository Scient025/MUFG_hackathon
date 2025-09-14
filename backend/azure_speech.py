import asyncio
import logging
import os
from typing import Any, Dict, Optional

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from project root (optional)
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    try:
        load_dotenv(env_path, encoding="utf-8")
    except (UnicodeDecodeError, ValueError):
        # Skip loading if file is corrupted
        print(
            f"Warning: Could not load .env file at {env_path} - using system environment variables"
        )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureSpeechService:
    def __init__(self):
        # Load Azure Speech configuration from environment variables
        self.speech_key = os.getenv("AZURE_SPEECH_KEY", "")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION", "")

        if not self.speech_key or not self.speech_region:
            logger.warning(
                "Azure Speech Services not configured. TTS/STT features will be disabled."
            )
            self.enabled = False
        else:
            self.enabled = True
            # Configure speech service
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key, region=self.speech_region
            )
            logger.info("Azure Speech Services configured successfully")

    def text_to_speech(
        self, text: str, voice_name: str = "en-AU-NatashaNeural"
    ) -> Optional[bytes]:
        """Convert text to speech and return audio data"""
        if not self.enabled:
            logger.warning("Azure Speech Services not enabled")
            return None

        try:
            # Set voice
            self.speech_config.speech_synthesis_voice_name = voice_name

            # Create synthesizer
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config, audio_config=None
            )

            # Synthesize speech
            result = synthesizer.speak_text_async(text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info("Text-to-speech conversion successful")
                return result.audio_data
            else:
                logger.error(f"Speech synthesis failed: {result.reason}")
                return None

        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return None

    def speech_to_text(
        self, audio_data: bytes, language: str = "en-AU"
    ) -> Optional[str]:
        """Convert speech to text"""
        if not self.enabled:
            logger.warning("Azure Speech Services not enabled")
            return None

        try:
            # Configure speech recognition
            self.speech_config.speech_recognition_language = language

            # Create audio config from bytes
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=False)

            # Create recognizer
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config, audio_config=audio_config
            )

            # Perform recognition
            result = recognizer.recognize_once_async().get()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                logger.info("Speech-to-text conversion successful")
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning("No speech could be recognized")
                return None
            else:
                logger.error(f"Speech recognition failed: {result.reason}")
                return None

        except Exception as e:
            logger.error(f"Error in speech-to-text: {e}")
            return None

    def get_available_voices(self) -> Dict[str, Any]:
        """Get list of available voices"""
        if not self.enabled:
            return {"error": "Azure Speech Services not enabled"}

        try:
            # Common voices for different languages
            voices = {
                "en-AU": [
                    {
                        "name": "en-AU-NatashaNeural",
                        "display": "Natasha (Female, Australian English)",
                    },
                    {
                        "name": "en-AU-KenNeural",
                        "display": "Ken (Male, Australian English)",
                    },
                ],
                "en-US": [
                    {
                        "name": "en-US-AriaNeural",
                        "display": "Aria (Female, US English)",
                    },
                    {
                        "name": "en-US-DavisNeural",
                        "display": "Davis (Male, US English)",
                    },
                ],
                "en-GB": [
                    {
                        "name": "en-GB-SoniaNeural",
                        "display": "Sonia (Female, British English)",
                    },
                    {
                        "name": "en-GB-RyanNeural",
                        "display": "Ryan (Male, British English)",
                    },
                ],
            }
            return voices
        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            return {"error": str(e)}

    def create_audio_response(
        self, text: str, voice_name: str = "en-AU-NatashaNeural"
    ) -> Dict[str, Any]:
        """Create audio response for API"""
        audio_data = self.text_to_speech(text, voice_name)

        if audio_data:
            return {
                "success": True,
                "audio_data": audio_data.hex(),  # Convert to hex string for JSON
                "voice_used": voice_name,
                "text": text,
            }
        else:
            return {"success": False, "error": "Failed to generate audio", "text": text}

    def process_audio_input(
        self, audio_data: bytes, language: str = "en-AU"
    ) -> Dict[str, Any]:
        """Process audio input and return text"""
        text = self.speech_to_text(audio_data, language)

        if text:
            return {"success": True, "text": text, "language": language}
        else:
            return {
                "success": False,
                "error": "Failed to recognize speech",
                "language": language,
            }


# Global Azure Speech Service instance
azure_speech_service = AzureSpeechService()


# Pydantic models for API
class TTSRequest(BaseModel):
    text: str
    voice_name: Optional[str] = "en-AU-NatashaNeural"


class STTRequest(BaseModel):
    audio_data: str  # Hex encoded audio data
    language: Optional[str] = "en-AU"


class VoiceListResponse(BaseModel):
    voices: Dict[str, Any]
    enabled: bool
