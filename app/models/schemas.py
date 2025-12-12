from pydantic import BaseModel
from typing import Literal, List

# --- LLM Schemas ---

# Define the literal types for emotions for strict validation.
Emotion = Literal[
    'curious',
    'skeptical',
    'interested',
    'frustrated',
    'impressed',
    'neutral'
]

class LLMResponse(BaseModel):
    """
    Pydantic model for validating the JSON structure of the LLM's response.
    This ensures the LLM output is predictable and can be safely processed.
    """
    customer_utterance: str
    customer_emotion: Emotion

# --- WebSocket Message Schemas ---

class WebSocketMessage(BaseModel):
    """
    Base model for any message sent over the WebSocket.
    The 'type' field allows the client to know how to handle the message.
    """
    type: str
    data: dict

class TranscriptData(BaseModel):
    """
    The 'data' part of a WebSocket message of type 'transcript'.
    """
    text: str
    is_final: bool

class ErrorData(BaseModel):
    """
    The 'data' part of a WebSocket message of type 'error'.
    """
    message: str

# --- Conversation Log Schemas ---

class ConversationTurn(BaseModel):
    """
    Represents a single turn in the conversation (one user utterance and one AI response).
    """
    user_utterance: str
    ai_response: LLMResponse
    timestamp: str

class SessionLog(BaseModel):
    """
    Represents the complete log for a single training session.
    """
    session_id: str
    start_time: str
    end_time: str
    conversation_history: List[ConversationTurn] = []