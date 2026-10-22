import asyncio
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import GetBusinessConnection, ReadBusinessMessage, SendChatAction, SendMessage
from aiogram.types import BusinessConnection, Message, Update


os.environ.setdefault('AISUP_DATABASE_URL', 'sqlite+aiosqlite:///./data/ai_support_test.db')
os.environ.setdefault('AISUP_MAIN_DATABASE_URL', '')
os.environ.setdefault('AISUP_INCLUDE_REMNAWAVE_DATA', 'false')

from ai_support_bot.app.bot import bot as bot_mod
from ai_support_bot.app.services.openai_client import OpenAIError


@pytest.fixture
def service(monkeypatch):
    bot_mod._processed_updates.clear()
    bot_mod._last_message_at.clear()
    bot_mod._throttle_warned.clear()
    bot = Bot('123456:TEST_TOKEN')
    connection = BusinessConnection(
        id='business-1',
        user={'id': 900, 'is_bot': False, 'first_name': 'Owner'},
        user_chat_id=900,
        date=datetime.now(UTC),
        is_enabled=True,
        rights={'can_reply': True, 'can_read_messages': True},
    )
    requests = []

    async def request(bot, method, **kwargs):
        requests.append(method)
        if isinstance(method, GetBusinessConnection):
            return connection
        if isinstance(method, (SendChatAction, ReadBusinessMessage)):
            return True
        return Message(
            message_id=100,
            date=datetime.now(UTC),
            chat={'id': method.chat_id, 'type': 'private'},
            text=method.text,
        )

    monkeypatch.setattr(bot.session, 'make_request', request)

    @asynccontextmanager
    async def session():
        yield None

    async def generate(*args, **kwargs):
        await asyncio.sleep(0)  # Let the typing task send its action.
        return {'answer': 'Ответ поддержки', 'escalate': False}

    agent = AsyncMock(side_effect=generate)
    monkeypatch.setattr(bot_mod, 'AsyncSessionLocal', session)
    monkeypatch.setattr(bot_mod.support_agent, 'generate_answer', agent)
    monkeypatch.setattr(bot_mod.settings_store, 'load', AsyncMock())
    monkeypatch.setattr(bot_mod.settings_store, 'get_int', lambda key: 0)
    monkeypatch.setattr(bot_mod.settings_store, 'get_bool', lambda key: False)
    monkeypatch.setattr(bot_mod.settings, 'ADMIN_IDS', '999')
    yield bot, connection, requests, agent
    bot_mod._processed_updates.clear()
    bot_mod._last_message_at.clear()
    bot_mod._throttle_warned.clear()


def message(**changes):
    values = dict(
        message_id=1,
        date=datetime.now(UTC),
        chat={'id': 555, 'type': 'private'},
        from_user={'id': 555, 'is_bot': False, 'first_name': 'Customer'},
        business_connection_id='business-1',
        text='Не работает VPN',
    )
    values.update(changes)
    return Message(**values)


async def dispatch(bot, msg, business=True):
    update = Update(update_id=1, **{'business_message' if business else 'message': msg})
    await bot_mod.build_dispatcher().feed_update(bot, update)


@pytest.mark.asyncio
async def test_business_answer_and_typing_use_connection_and_customer_identity(service):
    bot, _, requests, agent = service
    await dispatch(bot, message())
    assert agent.call_args.args[1:3] == (555, 'Не работает VPN')
    sent = [r for r in requests if isinstance(r, (SendMessage, SendChatAction))]
    assert len(sent) == 2
    assert all(r.chat_id == 555 and r.business_connection_id == 'business-1' for r in sent)
    reads = [r for r in requests if isinstance(r, ReadBusinessMessage)]
    assert len(reads) == 1
    assert (reads[0].business_connection_id, reads[0].chat_id, reads[0].message_id) == ('business-1', 555, 1)
    assert requests.index(reads[0]) > next(i for i, r in enumerate(requests) if isinstance(r, SendMessage))


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['disabled', 'no_rights', 'no_reply', 'owner', 'bot', 'sent_by_bot', 'no_sender'])
async def test_business_ignores_ineligible_messages(service, monkeypatch, case):
    bot, connection, requests, agent = service
    changes = {}
    if case == 'disabled':
        connection = connection.model_copy(update={'is_enabled': False})
    elif case == 'no_rights':
        connection = connection.model_copy(update={'rights': None})
    elif case == 'no_reply':
        connection = connection.model_copy(update={'rights': connection.rights.model_copy(update={'can_reply': False})})
    elif case == 'owner':
        changes['from_user'] = connection.user
    elif case == 'bot':
        changes['from_user'] = {'id': 777, 'is_bot': True, 'first_name': 'Bot'}
    elif case == 'sent_by_bot':
        changes['sender_business_bot'] = {'id': 777, 'is_bot': True, 'first_name': 'Bot'}
    elif case == 'no_sender':
        changes['from_user'] = None
    monkeypatch.setattr(bot, 'get_business_connection', AsyncMock(return_value=connection))
    await dispatch(bot, message(**changes))
    agent.assert_not_called()
    assert not any(isinstance(r, SendMessage) for r in requests)


@pytest.mark.asyncio
async def test_connection_checked_again_after_permission_revoked(service, monkeypatch):
    bot, connection, requests, agent = service
    await dispatch(bot, message())
    connection = connection.model_copy(update={'rights': connection.rights.model_copy(update={'can_reply': False})})
    lookup = AsyncMock(return_value=connection)
    monkeypatch.setattr(bot, 'get_business_connection', lookup)
    await dispatch(bot, message(message_id=2))
    assert agent.call_count == 1
    lookup.assert_awaited_once_with('business-1')


@pytest.mark.asyncio
async def test_connection_lookup_failure_does_not_answer_as_regular_bot(service, monkeypatch):
    bot, _, requests, agent = service
    monkeypatch.setattr(
        bot,
        'get_business_connection',
        AsyncMock(
            side_effect=TelegramBadRequest(
                method=GetBusinessConnection(business_connection_id='business-1'),
                message='Not found',
            )
        ),
    )
    await dispatch(bot, message())
    agent.assert_not_called()
    assert requests == []


@pytest.mark.asyncio
async def test_business_escalation_admin_notification_uses_regular_bot(service):
    bot, _, requests, agent = service
    agent.side_effect = None
    agent.return_value = {'answer': 'Передаю оператору', 'escalate': True}
    await dispatch(bot, message())
    sent = [r for r in requests if isinstance(r, SendMessage)]
    assert [(r.chat_id, r.business_connection_id) for r in sent] == [(555, 'business-1'), (999, None)]
    assert not any(isinstance(r, ReadBusinessMessage) for r in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['photo_disabled', 'generation_error', 'throttle'])
async def test_business_validation_and_error_replies_keep_connection(service, monkeypatch, case):
    bot, _, requests, agent = service
    msg = message()
    if case == 'photo_disabled':
        msg = message(text=None, photo=[{'file_id': 'photo', 'file_unique_id': 'unique', 'width': 10, 'height': 10}])
    elif case == 'generation_error':
        agent.side_effect = OpenAIError('Unavailable')
    else:
        monkeypatch.setattr(bot_mod, '_throttle_hit', lambda user_id: True)
    await dispatch(bot, msg)
    sent = [r for r in requests if isinstance(r, SendMessage)]
    assert len(sent) == 1
    assert sent[0].chat_id == 555
    assert sent[0].business_connection_id == 'business-1'
    assert not any(isinstance(r, ReadBusinessMessage) for r in requests)


@pytest.mark.asyncio
async def test_duplicate_detection_separates_regular_and_business_chats(service):
    bot, _, requests, agent = service
    await dispatch(bot, message())
    await dispatch(bot, message())
    await dispatch(bot, message(business_connection_id=None), business=False)
    await dispatch(bot, message(business_connection_id='business-2'))
    assert agent.call_count == 3
    sent = [r for r in requests if isinstance(r, SendMessage)]
    assert [r.business_connection_id for r in sent] == ['business-1', None, 'business-2']


@pytest.mark.asyncio
async def test_connection_updates_registered(service):
    bot, connection, _, agent = service
    dp = bot_mod.build_dispatcher()
    assert {'message', 'business_message', 'business_connection'} <= set(dp.resolve_used_update_types())
    await dp.feed_update(bot, Update(update_id=1, business_connection=connection))
    agent.assert_not_called()


@pytest.mark.asyncio
async def test_escalation_mid_dialogue_leaves_last_message_unread(service):
    bot, _, requests, agent = service
    await dispatch(bot, message(message_id=1))
    agent.side_effect = None
    agent.return_value = {'answer': 'Передаю оператору', 'escalate': True}
    await dispatch(bot, message(message_id=2, text='Позовите оператора'))
    assert [r.message_id for r in requests if isinstance(r, ReadBusinessMessage)] == [1]
    assert agent.call_count == 2


@pytest.mark.asyncio
async def test_answer_without_read_permission_still_delivered(service, monkeypatch):
    bot, connection, requests, _ = service
    connection = connection.model_copy(
        update={'rights': connection.rights.model_copy(update={'can_read_messages': False})}
    )
    monkeypatch.setattr(bot, 'get_business_connection', AsyncMock(return_value=connection))
    await dispatch(bot, message())
    assert any(isinstance(r, SendMessage) for r in requests)
    assert not any(isinstance(r, ReadBusinessMessage) for r in requests)


@pytest.mark.asyncio
async def test_read_failure_does_not_repeat_answer_or_escalate(service, monkeypatch):
    bot, _, requests, _ = service
    read = AsyncMock(
        side_effect=TelegramBadRequest(
            method=ReadBusinessMessage(business_connection_id='business-1', chat_id=555, message_id=1),
            message='Permission revoked',
        )
    )
    monkeypatch.setattr(bot, 'read_business_message', read)
    await dispatch(bot, message())
    read.assert_awaited_once_with(business_connection_id='business-1', chat_id=555, message_id=1)
    assert [r.chat_id for r in requests if isinstance(r, SendMessage)] == [555]


@pytest.mark.asyncio
async def test_failed_answer_delivery_does_not_mark_message_read(service, monkeypatch):
    bot, _, requests, _ = service
    monkeypatch.setattr(
        bot_mod,
        '_answer_safely',
        AsyncMock(
            side_effect=TelegramBadRequest(
                method=SendMessage(chat_id=555, text='Ответ'),
                message='Chat unavailable',
            )
        ),
    )
    with pytest.raises(TelegramBadRequest):
        await dispatch(bot, message())
    assert not any(isinstance(r, ReadBusinessMessage) for r in requests)


@pytest.mark.asyncio
async def test_regular_chat_does_not_use_business_read_api(service):
    bot, _, requests, _ = service
    await dispatch(bot, message(business_connection_id=None), business=False)
    assert any(isinstance(r, SendMessage) for r in requests)
    assert not any(isinstance(r, ReadBusinessMessage) for r in requests)
