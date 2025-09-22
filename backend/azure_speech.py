import asyncio
import logging
import os
from typing import Any, Dict, Optional

import azure.cognitiveservices.speech as speechsdk
import requests
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
                # Fallback to REST
                return self._rest_text_to_speech(text, voice_name)

        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            # Fallback to REST pathway if SDK fails to initialize
            try:
                return self._rest_text_to_speech(text, voice_name)
            except Exception as rest_err:
                logger.error(f"REST TTS fallback failed: {rest_err}")
                return None

    def _rest_text_to_speech(self, text: str, voice_name: str) -> Optional[bytes]:
        """Fallback TTS using Azure REST API (avoids native SDK deps)."""
        if not self.speech_key or not self.speech_region:
            return None
        # Get access token
        token_url = f"https://{self.speech_region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
        headers = {"Ocp-Apim-Subscription-Key": self.speech_key, "Content-Length": "0"}
        resp = requests.post(token_url, headers=headers, timeout=10)
        resp.raise_for_status()
        access_token = resp.text

        # Build SSML
        ssml = f"""
        <speak version='1.0' xml:lang='en-AU'>
          <voice name='{voice_name}'>
            {text}
          </voice>
        </speak>
        """.strip()

        synthesis_url = f"https://{self.speech_region}.tts.speech.microsoft.com/cognitiveservices/v1"
        synth_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/ssml+xml",
            # WAV/RIFF PCM 16kHz mono 16-bit to match frontend expectation
            "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
            "User-Agent": "mufg-hackathon-app",
        }
        synth_resp = requests.post(
            synthesis_url, headers=synth_headers, data=ssml.encode("utf-8"), timeout=30
        )
        synth_resp.raise_for_status()
        return synth_resp.content

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

            # Create audio stream from bytes using PushAudioInputStream
            audio_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=16000, bits_per_sample=16, channels=1
            )

            # Create push audio input stream
            audio_stream = speechsdk.audio.PushAudioInputStream(audio_format)

            # Write audio data to stream
            audio_stream.write(audio_data)
            audio_stream.close()

            # Create audio config from stream
            audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)

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
        logger.info(f"Processing audio input: {len(audio_data)} bytes")
        text = self.speech_to_text(audio_data, language)

        if text:
            logger.info(f"Recognized text: {text}")
            return {"success": True, "text": text, "language": language}
        else:
            logger.warning("No text recognized from audio")
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
