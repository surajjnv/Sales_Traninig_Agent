import logging
from google.cloud import texttospeech

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToSpeechService:
    """
    Manages the synthesis of text into speech using Google Cloud TTS.
    """

    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()

    async def synthesize_speech(self, text: str) -> bytes:
        """
        Synthesizes the given text into audio bytes.

        Args:
            text: The text to be synthesized.

        Returns:
            The audio content in bytes (e.g., MP3 data).
        """
        try:
            logger.info(f"Synthesizing speech for text: '{text}'")

            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Configure the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code=settings.LANGUAGE_CODE,
                name=settings.TTS_VOICE_NAME
            )

            # Configure the audio format
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding[settings.TTS_AUDIO_ENCODING]
            )

            # Perform the text-to-speech request
            response = self.client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            logger.info("Speech synthesis successful.")
            return response.audio_content

        except Exception as e:
            logger.error(f"An error occurred during speech synthesis: {e}", exc_info=True)
            # Return empty bytes or handle the error as appropriate
            return b''