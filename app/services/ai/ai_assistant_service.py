from datetime import UTC, datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database.crud.ai_conversation import AIConversationCRUD
from app.database.models import AIConversation, User
from app.services.ai.knowledge_base_service import KnowledgeBaseService
from app.services.ai.llm_client import LLMClient, LLMError
from app.services.ai.user_context import build_user_context


logger = structlog.get_logger(__name__)

HANDOFF_TAG = '[HANDOFF]'


class AIReply:
    __slots__ = ('error', 'handoff', 'limited', 'text')

    def __init__(self, text: str, *, handoff: bool = False, limited: bool = False, error: bool = False):
        self.text = text
        self.handoff = handoff
        self.limited = limited
        self.error = error


class AIAssistantService:
    """Ядро ИИ-ассистента поддержки.

    Отвечает за построение экономного контекста, вызов LLM, скользящую
    суммаризацию переписки каждые N сообщений и передачу диалога оператору.
    """

    @staticmethod
    def is_available() -> bool:
        return settings.is_ai_assistant_enabled()

    @classmethod
    def _system_prompt(cls, company: str, bot_username: str) -> str:
        bot_ref = f'@{bot_username}' if bot_username else 'наш бот'
        return (
            f'Ты — ИИ-помощник службы поддержки сервиса {company} (VPN по подписке через Telegram-бота).\n'
            f'Твоя задача — быстро и по делу решать вопросы клиентов и снижать нагрузку на живых операторов.\n\n'
            'ПРАВИЛА ОБЩЕНИЯ:\n'
            '- Пиши на языке клиента, по умолчанию на русском. Тон дружелюбный, тёплый, вежливый, можно немного эмодзи.\n'
            '- Отвечай кратко и структурировано: если это инструкция — нумерованные шаги, без воды.\n'
            '- Опирайся на блок «БАЗА ЗНАНИЙ» и «ДАННЫЕ КЛИЕНТА». Не выдумывай фактов, которых там нет.\n'
            '- Используй данные клиента, чтобы отвечать персонально (баланс, подписка, срок действия и т.п.).\n'
            '- На бытовые вопросы («как дела», «привет», «спасибо») отвечай коротко и по-человечески, '
            'затем мягко предложи помощь по VPN.\n'
            f'- Все действия с подпиской, оплатой, устройствами клиент делает в боте {bot_ref}.\n'
            '- Не запрашивай пароли, коды и платёжные данные. Не обещай возвраты и индивидуальные условия от лица компании.\n\n'
            'КОГДА ПЕРЕДАВАТЬ ОПЕРАТОРУ:\n'
            '- Если вопрос требует действий на стороне компании (проверка платежа, возврат, разблокировка, '
            'индивидуальные/партнёрские условия, сбой на сервере), либо клиент прямо просит оператора, '
            'либо ты не можешь помочь по базе знаний.\n'
            f'- В этом случае в САМОМ КОНЦЕ ответа добавь отдельной строкой тег {HANDOFF_TAG}. '
            'Клиенту тег не показывай по смыслу — просто напиши, что передаёшь вопрос оператору, и добавь тег в конце.\n'
            '- Если справляешься сам — тег не добавляй.'
        )

    @classmethod
    def _clip(cls, text: str, limit: int) -> str:
        text = (text or '').strip()
        if len(text) > limit:
            return text[:limit].rstrip() + '…'
        return text

    @classmethod
    def _reset_daily_if_needed(cls, conversation: AIConversation) -> None:
        today = datetime.now(UTC).strftime('%Y-%m-%d')
        if conversation.day_bucket != today:
            conversation.day_bucket = today
            conversation.messages_today = 0

    @classmethod
    async def _summarize(cls, conversation: AIConversation) -> None:
        """Сжимает старую переписку в короткое саммари для экономии токенов."""
        history = conversation.recent_messages or []
        if not history:
            return

        dialogue_lines = []
        for item in history:
            role = 'Клиент' if item.get('role') == 'user' else 'Ассистент'
            dialogue_lines.append(f'{role}: {item.get("content", "")}')
        dialogue = '\n'.join(dialogue_lines)

        prompt = [
            {
                'role': 'system',
                'content': (
                    'Ты сжимаешь диалог поддержки в краткое саммари на русском (до 4 предложений). '
                    'Сохрани суть проблемы клиента, что уже выяснено/сделано и что осталось. '
                    'Без приветствий и воды, только факты для продолжения диалога.'
                ),
            },
        ]
        if conversation.summary:
            prompt.append({'role': 'user', 'content': f'Предыдущее саммари:\n{conversation.summary}'})
        prompt.append({'role': 'user', 'content': f'Диалог:\n{dialogue}'})

        try:
            summary = await LLMClient.chat(
                prompt,
                model=settings.get_ai_assistant_summary_model(),
                temperature=0.2,
                max_tokens=settings.AI_ASSISTANT_SUMMARY_MAX_TOKENS,
            )
        except LLMError as error:
            logger.warning('Summary generation failed', error=str(error))
            return

        if summary:
            conversation.summary = cls._clip(summary, 1500)
            conversation.recent_messages = history[-2:]

    @classmethod
    async def process_message(cls, db: AsyncSession, user: User, text: str) -> AIReply:
        if not cls.is_available():
            return AIReply('', error=True)

        text = cls._clip(text, settings.AI_ASSISTANT_MAX_INPUT_CHARS)
        if not text:
            return AIReply('Напишите, пожалуйста, ваш вопрос текстом 🙂')

        conversation = await AIConversationCRUD.get_or_create(db, user.id)
        cls._reset_daily_if_needed(conversation)

        limit = settings.AI_ASSISTANT_DAILY_MESSAGE_LIMIT or 0
        if limit and conversation.messages_today >= limit:
            return AIReply('', limited=True)

        company = settings.AI_ASSISTANT_COMPANY_NAME or 'наш сервис'
        bot_username = settings.get_ai_assistant_bot_username()

        kb_context = KnowledgeBaseService.build_context(text)
        user_context = await build_user_context(db, user)

        messages: list[dict] = [
            {'role': 'system', 'content': cls._system_prompt(company, bot_username)},
        ]
        context_block = f'ДАННЫЕ КЛИЕНТА:\n{user_context}'
        if conversation.summary:
            context_block += f'\n\nКРАТКОЕ СОДЕРЖАНИЕ ПРЕДЫДУЩЕГО ДИАЛОГА:\n{conversation.summary}'
        if kb_context:
            context_block += f'\n\nБАЗА ЗНАНИЙ (используй для ответа):\n{kb_context}'
        messages.append({'role': 'system', 'content': context_block})

        history = conversation.recent_messages or []
        history_limit = max(0, settings.AI_ASSISTANT_HISTORY_LIMIT)
        for item in history[-history_limit:]:
            messages.append({'role': item.get('role', 'user'), 'content': item.get('content', '')})
        messages.append({'role': 'user', 'content': text})

        try:
            answer = await LLMClient.chat(messages)
        except LLMError as error:
            logger.error('AI assistant reply failed', error=str(error), user_id=user.id)
            return AIReply('', error=True)

        handoff = HANDOFF_TAG in answer
        clean_answer = answer.replace(HANDOFF_TAG, '').strip()
        if not clean_answer:
            clean_answer = 'Секунду, передаю ваш вопрос оператору 🙌'

        new_history = list(history)
        new_history.append({'role': 'user', 'content': text})
        new_history.append({'role': 'assistant', 'content': cls._clip(clean_answer, 1200)})
        conversation.recent_messages = new_history
        conversation.user_message_count = (conversation.user_message_count or 0) + 1
        conversation.total_messages = (conversation.total_messages or 0) + 1
        conversation.messages_today = (conversation.messages_today or 0) + 1
        conversation.last_message_at = datetime.now(UTC)
        if handoff:
            conversation.handed_off = True

        every = max(1, settings.AI_ASSISTANT_SUMMARIZE_EVERY)
        if conversation.user_message_count % every == 0:
            await cls._summarize(conversation)

        await db.flush()

        return AIReply(clean_answer, handoff=handoff)

    @classmethod
    async def reset(cls, db: AsyncSession, user_id: int) -> None:
        await AIConversationCRUD.reset(db, user_id)
        await db.flush()
