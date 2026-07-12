"""
Milestone 3: Local Llama 3.2 agent via Ollama.
Sends transcription text with a persona prompt and returns the agent's response.
"""
from __future__ import annotations

DEFAULT_PERSONA = (
    "You are Ali Pilehvar Meibody, Iranian, 26 years old, "
    "Material Engineering student at Politecnico di Torino. "
    "Answer  questions as Ali would and  short in one line"
)

MODEL = "llama3.2:latest"
MAX_INPUT_CHARS = 2000  # Limit query size to avoid overloading the local model


def get_agent_response(
    new_text: str,
    persona: str = DEFAULT_PERSONA,
    max_chars: int = MAX_INPUT_CHARS,
) -> str:
    """
    Send new transcription text to local Llama 3.2 via Ollama with persona.
    Runs synchronously; call from a background thread to keep UI responsive.
    """
    new_text = (new_text or "").strip()
    if not new_text:
        return ""
    if len(new_text) > max_chars:
        new_text = new_text[-max_chars:].strip()
    prompt = f"""[Persona]:
{persona}

[Conversation]:
{new_text}

Respond as the persona above."""
    try:
        from ollama import chat
        response = chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        msg = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
        if msg is None:
            return "[Agent: no response]"
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        return (content or "").strip() or "[Agent: empty response]"
    except Exception as e:
        return f"[Agent error: {e}]"
