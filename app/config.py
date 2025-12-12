import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from a .env file
# The .env file is expected to be in the 'backend' directory
# This script is in 'backend/app', so we go up one level.
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path)

class Settings(BaseSettings):
    """
    Pydantic model to manage application settings.
    It automatically reads environment variables.
    """
    # Google Cloud a
    # nd Gemini Configuration
    GOOGLE_APPLICATION_CREDENTIALS: str
    LLM_API_KEY: str
    
    # LLM Persona Configuration
    CUSTOMER_SYSTEM_PROMPT: str
    
    # Websocket and Audio Configuration
    # Audio encoding for Google STT
    AUDIO_ENCODING: str = 'AUDIO_ENCODING_LINEAR_16'
    # Sample rate of the audio from the client
    SAMPLE_RATE_HERTZ: int = 16000
    # BCP-47 language code for STT
    LANGUAGE_CODE: str = 'en-US'

    # Google TTS Configuration
    # Voice name for the TTS response
    TTS_VOICE_NAME: str = 'en-US-Wavenet-D'
    # TTS Audio encoding
    TTS_AUDIO_ENCODING: str = 'MP3'


# Create a single settings instance to be used throughout the application
settings = Settings()

# You can add a simple check to ensure credentials file exists
if not os.path.exists(settings.GOOGLE_APPLICATION_CREDENTIALS):
    # Note: In a real app, you might want a more robust check or logging
    print(f"WARNING: GOOGLE_APPLICATION_CREDENTIALS file not found at '{settings.GOOGLE_APPLICATION_CREDENTIALS}'")
    # In production, you might want to raise an exception:
    # raise FileNotFoundError(f"Service account key file not found at {settings.GOOGLE_APPLICATION_CREDENTIALS}")