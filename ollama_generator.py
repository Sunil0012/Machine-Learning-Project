# ollama_generator.py
import requests
import json

class OllamaResponseGenerator:
    def __init__(self, model_name="llama3:8b", api_url=None):
        self.model_name = model_name
        self.api_url = api_url or "http://localhost:11434/api/generate"
        self.default_timeout = 120  # Increased from 10 to 120 seconds for LLM responses

    def generate_response(self, user_input, emotion_hint=None):
        """
        Generate an empathetic response using Ollama.
        
        Args:
            user_input: The user's message
            emotion_hint: Optional emotion context
            
        Returns:
            Generated response string
        """
        # Build prompt with emotion context if provided
        if emotion_hint:
            prompt = f"""You are an empathetic emotional-support chatbot.
User emotion: {emotion_hint}
User says: "{user_input}"

Respond with compassion, emotional warmth, and helpful guidance."""
        else:
            prompt = f"""You are an empathetic emotional-support chatbot.
User says: "{user_input}"

Respond with compassion, emotional warmth, and helpful guidance."""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            resp = requests.post(
                self.api_url, 
                json=payload, 
                headers=headers, 
                timeout=self.default_timeout
            )
            resp.raise_for_status()
            response_json = resp.json()
            
            # Handle Ollama's response format
            if isinstance(response_json, dict):
                # Standard Ollama format uses "response" key
                if "response" in response_json:
                    return str(response_json["response"]).strip()
                
                # Try other common keys as fallback
                for key in ("result", "generated", "text", "output"):
                    if key in response_json:
                        val = response_json[key]
                        if isinstance(val, str):
                            return val.strip()
                        if isinstance(val, dict) and "text" in val:
                            return str(val["text"]).strip()
                
                # If no recognized key, stringify the whole response
                return json.dumps(response_json)
            else:
                return str(response_json)
                
        except requests.exceptions.Timeout:
            return "(Error: Request timed out)"
        except requests.exceptions.ConnectionError:
            return "(Error: Could not connect to Ollama. Is it running?)"
        except requests.exceptions.HTTPError as e:
            return f"(Error: HTTP {e.response.status_code})"
        except Exception as e:
            return f"(Error: {str(e)})"