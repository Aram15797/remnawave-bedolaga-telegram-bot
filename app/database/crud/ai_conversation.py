from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import AIConversation


logger = structlog.get_logger(__name__)


class AIConversationCRUD:
    """CRUD для состояния диалога пользователя с ИИ-ассистентом."""

    @staticmethod
    async def get_by_user(db: AsyncSession, user_id: int) -> AIConversation | None:
        result = await db.execute(select(AIConversation).where(AIConversation.user_id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_or_create(db: AsyncSession, user_id: int) -> AIConversation:
        conversation = await AIConversationCRUD.get_by_user(db, user_id)
        if conversation:
            return conversation
        conversation = AIConversation(
            user_id=user_id,
            summary='',
            recent_messages=[],
            user_message_count=0,
            total_messages=0,
            messages_today=0,
            day_bucket=datetime.now(UTC).strftime('%Y-%m-%d'),
            handed_off=False,
        )
        db.add(conversation)
        await db.flush()
        return conversation

    @staticmethod
    async def reset(db: AsyncSession, user_id: int) -> None:
        conversation = await AIConversationCRUD.get_by_user(db, user_id)
        if not conversation:
            return
        conversation.summary = ''
        conversation.recent_messages = []
        conversation.user_message_count = 0
        conversation.total_messages = 0
        conversation.handed_off = False
        await db.flush()
