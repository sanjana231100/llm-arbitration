import instructor
from groq import Groq
from openai import OpenAI
from pydantic import BaseModel
from typing import TypeVar
from .config import get_settings

T = TypeVar("T", bound=BaseModel)

CRITIC_MODELS = {
    "factual": "llama-3.3-70b-versatile",
    "logical": "mixtral-8x7b-32768",
    "completeness": "gemma2-9b-it",
    "adjudicator": "llama-3.3-70b-versatile",
}


def _groq_client() -> instructor.Instructor:
    settings = get_settings()
    return instructor.from_groq(Groq(api_key=settings.groq_api_key))


def _ollama_client() -> instructor.Instructor:
    settings = get_settings()
    return instructor.from_openai(
        OpenAI(
            base_url=f"{settings.ollama_base_url}/v1",
            api_key="ollama",
        )
    )


def call_model(
    critic: str,
    system_prompt: str,
    user_message: str,
    response_model: type[T],
) -> T:
    settings = get_settings()

    if settings.groq_api_key:
        try:
            client = _groq_client()
            return client.chat.completions.create(
                model=CRITIC_MODELS[critic],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_model=response_model,
                max_retries=2,
            )
        except Exception as e:
            print(f"Groq failed for {critic}: {e} — falling back to Ollama")

    client = _ollama_client()
    return client.chat.completions.create(
        model=settings.ollama_fallback_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_model=response_model,
        max_retries=2,
    )
