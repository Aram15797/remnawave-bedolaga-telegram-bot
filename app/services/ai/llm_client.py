import httpx
import structlog

from app.config import settings


logger = structlog.get_logger(__name__)


class LLMError(Exception):
    """Ошибка обращения к LLM-провайдеру."""


class LLMClient:
    """Тонкий клиент для OpenAI-совместимого chat/completions API.

    Работает с OpenAI, DeepSeek, OpenRouter и любым совместимым эндпоинтом —
    достаточно указать BASE_URL, API_KEY и MODEL в настройках.
    """

    @staticmethod
    def _base_url() -> str:
        return (settings.AI_ASSISTANT_BASE_URL or 'https://api.openai.com/v1').rstrip('/')

    @classmethod
    async def chat(
        cls,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        api_key = (settings.AI_ASSISTANT_API_KEY or '').strip()
        if not api_key:
            raise LLMError('AI assistant API key is not configured')

        payload = {
            'model': model or settings.AI_ASSISTANT_MODEL,
            'messages': messages,
            'temperature': settings.AI_ASSISTANT_TEMPERATURE if temperature is None else temperature,
            'max_tokens': max_tokens if max_tokens is not None else settings.AI_ASSISTANT_MAX_TOKENS,
        }
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        url = f'{cls._base_url()}/chat/completions'
        timeout = httpx.Timeout(float(settings.AI_ASSISTANT_TIMEOUT_SECONDS or 40))

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as error:
            logger.error('LLM request failed', error=str(error))
            raise LLMError(f'LLM request failed: {error}') from error

        if response.status_code >= 400:
            logger.error('LLM returned error', status=response.status_code, body=response.text[:500])
            raise LLMError(f'LLM returned HTTP {response.status_code}')

        try:
            data = response.json()
            content = data['choices'][0]['message']['content']
        except (KeyError, IndexError, ValueError) as error:
            logger.error('Failed to parse LLM response', error=str(error), body=response.text[:500])
            raise LLMError('Failed to parse LLM response') from error

        return (content or '').strip()
