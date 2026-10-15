from argparse import Namespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select

from ai_support_bot import check_support
from ai_support_bot.app.db.models import Conversation


@pytest.mark.asyncio
async def test_live_check_reads_live_knowledge_but_stores_conversation_separately(monkeypatch, capsys):
    async with check_support.isolated_history() as live_session:
        live_engine = live_session.kw['bind']
        monkeypatch.setattr(check_support.database, 'AsyncSessionLocal', live_session)
        monkeypatch.setattr(check_support.database, 'engine', live_engine)
        monkeypatch.setattr(check_support.database, 'main_engine', None)
        monkeypatch.setattr(check_support.settings_store, 'load', AsyncMock())
        monkeypatch.setattr(check_support.settings_store, 'get', lambda key: 'test-model')

        async def retrieve(db, query):
            assert db.bind is live_engine
            return []

        async def generate(db, telegram_id, question):
            assert db.bind is not live_engine
            assert await check_support.rag_service.retrieve(db, question) == []
            db.add(Conversation(telegram_id=telegram_id))
            await db.commit()
            return {'answer': 'Передаю предложение команде.', 'escalate': True, 'knowledge_used': 0}

        monkeypatch.setattr(check_support.rag_service, 'retrieve', retrieve)
        monkeypatch.setattr(check_support.agent.support_agent, 'generate_answer', generate)
        await check_support.run(Namespace(telegram_id=42, email='qa@example.invalid', interactive=False, case='partnership'))
        async with live_session() as db:
            count = await db.scalar(select(func.count(Conversation.id)))
            assert count == 0
        assert 'Передаю предложение команде.' in capsys.readouterr().out
