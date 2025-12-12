import asyncio
import logging
from google.cloud.speech_v1.services.speech import SpeechAsyncClient as SpeechClient
from google.cloud.speech_v1.types import RecognitionConfig, StreamingRecognitionConfig, StreamingRecognizeRequest
from google.api_core import exceptions as google_exceptions

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechToTextService:
    """
    Manages the streaming speech-to-text transcription using Google Cloud STT.

    This service takes an audio queue and a callback. It continuously pulls audio
    chunks from the queue, sends them to Google STT, and invokes the callback
    with the transcription results.
    """
    def __init__(self, audio_queue: asyncio.Queue, on_transcript_update):
        self._audio_queue = audio_queue
        self._on_transcript_update = on_transcript_update
        self._speech_client = SpeechClient()

    async def _request_generator(self):
        """
        A generator that yields audio chunks from the queue to the Google STT API.
        It also yields the initial configuration message.
        """
        streaming_config = StreamingRecognitionConfig(
            config=RecognitionConfig(
                encoding=RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=settings.SAMPLE_RATE_HERTZ,
                language_code=settings.LANGUAGE_CODE,
                alternative_language_codes=settings.ALTERNATIVE_LANGUAGE_CODES,
                enable_automatic_punctuation=True,
            ),
            interim_results=True,
        )

        # The first request carries the configuration.
        yield StreamingRecognizeRequest(streaming_config=streaming_config)

        # Subsequent requests carry the audio data.
        while True:
            try:
                # Get a chunk of audio from the queue. This will block until a
                # chunk is available.
                chunk = await self._audio_queue.get()
                if chunk is None:
                    # A None chunk is a sentinel indicating the end of the stream.
                    break
                yield StreamingRecognizeRequest(audio_content=chunk)
            except asyncio.CancelledError:
                break

    async def process_audio_stream(self):
        """
        This is the main method of the service. It initiates the streaming
        transcription and handles the responses from Google STT.
        """
        try:
            requests = self._request_generator()
            # Use the async client for streaming recognition
            responses = await self._speech_client.streaming_recognize(requests=requests)

            logger.info("Google STT stream opened.")

            # Use 'async for' to iterate over the async generator
            async for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript
                is_final = result.is_final

                # Invoke the callback with the new transcript data.
                await self._on_transcript_update(transcript, is_final)

                if is_final:
                    logger.info(f"Final transcript received: {transcript}")

        except google_exceptions.OutOfRange as e:
            logger.info(f"Google STT stream timed out due to silence: {e}")
        except google_exceptions.Cancelled as e:
            logger.info(f"Google STT stream was cancelled: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in STT service: {e}", exc_info=True)
        finally:
            logger.info("Google STT stream closed.")