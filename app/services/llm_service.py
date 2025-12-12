import os
import logging
import json
from typing import List
from openai import OpenAI

from app.config import settings
from app.models.schemas import LLMResponse, ConversationTurn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomLLMClient:
    """Client for connecting to the custom LLM Gateway"""
    def __init__(self, api_key=None):
        """
        Initialize LLM Client
        Args:
            api_key: LLM Gateway API key. If not provided, will look for LLM_API_KEY environment variable
        """
        self.api_key = api_key
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://imllm.intermesh.net/v1"
        )

    def chat(self, prompt, model, system_message, temperature, max_tokens):
        """
        Send a chat completion request
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


class LLMService:
    """
    Manages interaction with the custom LLM Gateway.
    This service now uses the provided CustomLLMClient.
    """

    def __init__(self):
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY environment variable not found!")
        self.client = CustomLLMClient(api_key=api_key)

    def _format_prompt_with_history(self, user_utterance: str, history: List[ConversationTurn]) -> str:
        """
        Formats the conversation history and new utterance into a single string prompt.
        """
        formatted_history = "\n--- Conversation History ---\n"
        for turn in history:
            formatted_history += f"Trainee: {turn.user_utterance}\n"
            formatted_history += f"Customer (You): {turn.ai_response.model_dump_json()}\n"
        
        formatted_history += "--- New Utterance ---\n"
        formatted_history += f"Trainee: {user_utterance}\n"
        formatted_history += "Customer (You): "
        return formatted_history

    def _clean_json_string(self, json_string: str) -> str:
        """
        Cleans the raw output from the LLM to extract a valid JSON string.
        """
        start_index = json_string.find('{')
        end_index = json_string.rfind('}')
        if start_index == -1 or end_index == -1:
            logger.error(f"Could not find a valid JSON object in the string: {json_string}")
            return ""
        return json_string[start_index:end_index+1]

    async def generate_response(
        self, user_utterance: str, conversation_history: List[ConversationTurn]
    ) -> LLMResponse:
        """
        Generates a response from the LLM based on the user's utterance and history.
        """
        try:
            system_prompt = settings.CUSTOMER_SYSTEM_PROMPT
            full_user_prompt = self._format_prompt_with_history(user_utterance, conversation_history)

            logger.info("Sending request to custom LLM Gateway.")

            raw_text = self.client.chat(
                prompt=full_user_prompt,
                model="google/gemini-2.0-flash", # Using a model from the allowed list
                system_message=system_prompt,
                temperature=0.7,
                max_tokens=250
            )
            
            logger.info(f"Received raw response from LLM Gateway: {raw_text}")

            if not raw_text:
                raise ValueError("LLM returned an empty response.")

            cleaned_json_string = self._clean_json_string(raw_text)
            if not cleaned_json_string:
                raise ValueError("Failed to extract JSON from LLM response.")

            llm_response = LLMResponse.parse_raw(cleaned_json_string)
            
            logger.info(f"Successfully parsed LLM response: {llm_response}")
            return llm_response

        except Exception as e:
            logger.error(f"An error occurred while calling the LLM: {e}", exc_info=True)
            return LLMResponse(
                customer_utterance="I'm sorry, I'm having a little trouble right now. Can you repeat that?",
                customer_emotion='frustrated'
            )