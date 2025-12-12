import asyncio
import logging
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Import application modules
from app.services.stt_service import SpeechToTextService
from app.services.llm_service import LLMService
from app.services.tts_service import TextToSpeechService
from app.models.schemas import ConversationTurn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- CORS Middleware ---
# This allows the frontend (running on a different port/domain) to communicate with the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- WebSocket Endpoint ---
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for session: {session_id}")

    # --- Session State Initialization ---
    # Queue to hold incoming audio chunks from the client
    audio_queue = asyncio.Queue()
    # List to store the conversation history for the LLM context
    conversation_history: List[ConversationTurn] = []
    
    # --- Service Initialization ---
    llm_service = LLMService()
    tts_service = TextToSpeechService()

    # --- Core Callback: on_transcript_update ---
    # This function is called by the STT service with transcription updates.
    async def on_transcript_update(transcript: str, is_final: bool):
        # Send the transcript to the client
        await websocket.send_json({
            "type": "transcript",
            "data": {"text": transcript, "is_final": is_final}
        })

        # If the transcript is final, trigger the LLM and TTS flow
        if is_final:
            logger.info("Final transcript received. Processing with LLM and TTS.")

            # 1. Generate response from LLM
            llm_response = await llm_service.generate_response(transcript, conversation_history)
            
            # 2. Synthesize speech from LLM's text response
            audio_bytes = await tts_service.synthesize_speech(llm_response.customer_utterance)
            
            # 3. Send the synthesized audio back to the client
            if audio_bytes:
                # Encode audio bytes as Base64 to safely send via JSON
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                await websocket.send_json({
                    "type": "audio",
                    "data": audio_base64
                })
                logger.info("Sent TTS audio to client.")

            # 4. Update conversation history
            turn = ConversationTurn(
                user_utterance=transcript,
                ai_response=llm_response,
                timestamp=asyncio.get_event_loop().time().__str__()
            )
            conversation_history.append(turn)

    # Instantiate the STT service with the queue and callback
    stt_service = SpeechToTextService(audio_queue, on_transcript_update)

    # --- Main Tasks ---
    async def audio_receiver():
        """Receives audio chunks from the client and puts them in the queue."""
        while True:
            try:
                message = await websocket.receive_bytes()
                await audio_queue.put(message)
            except WebSocketDisconnect:
                logger.info("Client disconnected from audio_receiver.")
                # Put a sentinel value to signal the end of the audio stream
                await audio_queue.put(None)
                break

    stt_task = asyncio.create_task(stt_service.process_audio_stream())
    receiver_task = asyncio.create_task(audio_receiver())

    try:
        # Keep the connection alive by waiting for tasks to complete
        await asyncio.gather(stt_task, receiver_task)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"An error occurred in WebSocket endpoint: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaning up tasks for session: {session_id}")
        stt_task.cancel()
        receiver_task.cancel()
        # Ensure tasks are awaited to allow cleanup
        await asyncio.gather(stt_task, receiver_task, return_exceptions=True)
        logger.info("Connection closed.")

# --- Root Endpoint (for health checks) ---
@app.get("/")
async def root():
    return {"status": "alive"}