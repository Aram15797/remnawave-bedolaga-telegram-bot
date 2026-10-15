import asyncio
import json
import os
from unittest.mock import AsyncMock

from sqlalchemy import text

import pytest
import pytest_asyncio

os.environ.setdefault('AISUP_MAIN_DATABASE_URL', '')
os.environ.setdefault('AISUP_INCLUDE_REMNAWAVE_DATA', 'false')

from ai_support_bot.app.core.config import ESCALATION_MARKER


USER_CONTEXT = (
    'Telegram ID: 4242\n'
    'Баланс: 150.00 ₽\n'
    'Внимание: у пользователя 2 подписки.\n'
    'Подписки:\n'
    '  • подписка №1 (id=11), статус=active, до 01.10.2026, устройств=3\n'
    '  • подписка №2 (id=12), статус=active, до 05.11.2026, устройств=5'
)


@pytest_asyncio.fixture
async def flow(monkeypatch, tmp_path):
    db_file = tmp_path / 'flow.db'
    monkeypatch.setenv('AISUP_DATABASE_URL', f'sqlite+aiosqlite:///{db_file}')

    from importlib import reload

    from ai_support_bot.app.core import config as config_mod
    config_mod.get_settings.cache_clear()
    monkeypatch.setattr(config_mod, 'settings', config_mod.get_settings())

    from ai_support_bot.app.db import database as database_mod
    reload(database_mod)

    from ai_support_bot.app.services import agent as agent_mod
    from ai_support_bot.app.services import settings_store

    captured: dict = {'queries': [], 'prompts': [], 'reply': 'Готово.'}

    async def fake_chat_completion(messages, model, max_tokens, temperature, **kwargs):
        captured['prompts'].append(messages[0]['content'])
        captured['messages'] = messages
        result = {
            'content': captured['reply'],
            'model': model,
            'tokens_prompt': 1,
            'tokens_completion': 1,
            'tokens_cached': 0,
        }
        if captured.get('responses'):
            result.update(captured['responses'].pop(0))
        return result

    async def fake_retrieve(db, query):
        captured['queries'].append(query)
        return captured.get('knowledge', [])

    async def fake_user_context(telegram_id):
        return USER_CONTEXT

    monkeypatch.setattr(agent_mod.openai_client, 'chat_completion', fake_chat_completion)
    monkeypatch.setattr(agent_mod.rag_service, 'retrieve', fake_retrieve)
    monkeypatch.setattr(agent_mod, 'build_user_context', fake_user_context)
    monkeypatch.setattr(agent_mod, '_response_cache', agent_mod._ResponseCache())

    monkeypatch.setattr(settings_store, 'AsyncSessionLocal', database_mod.AsyncSessionLocal)
    monkeypatch.setattr(settings_store, 'settings', config_mod.settings)
    monkeypatch.setattr(settings_store, '_cache', {})
    monkeypatch.setattr(settings_store, '_DEFAULTS', {})

    await database_mod.init_db()
    await settings_store.load()
    monkeypatch.setitem(settings_store._cache, 'SUMMARY_ENABLED', '0')

    try:
        yield captured, agent_mod, database_mod.AsyncSessionLocal
    finally:
        await database_mod.engine.dispose()


@pytest.mark.asyncio
async def test_user_data_is_always_injected_into_prompt(flow):
    captured, agent_mod, Session = flow

    async with Session() as db:
        await agent_mod.support_agent.generate_answer(db, 4242, 'какой у меня лимит устройств')

    prompt = captured['prompts'][-1]
    assert 'Данные текущего пользователя:' in prompt
    assert 'подписка №1 (id=11)' in prompt
    assert 'подписка №2 (id=12)' in prompt
    assert 'ПРАВИЛО ИСПОЛЬЗОВАНИЯ ДАННЫХ' in prompt


@pytest.mark.asyncio
async def test_topic_switch_keeps_focus_rule_and_contextualizes_retrieval(flow):
    captured, agent_mod, Session = flow

    async with Session() as db:
        await agent_mod.support_agent.generate_answer(db, 4242, 'не проходит оплата картой')
    async with Session() as db:
        await agent_mod.support_agent.generate_answer(db, 4242, 'сколько устройств можно подключить')
    async with Session() as db:
        await agent_mod.support_agent.generate_answer(db, 4242, 'а для второй подписки как?')

    last_query = captured['queries'][-1]
    assert last_query.strip().endswith('Текущий вопрос: а для второй подписки как?')
    assert 'сколько устройств можно подключить' in last_query

    prompt = captured['prompts'][-1]
    assert 'ПРАВИЛО ФОКУСА' in prompt

    history_roles = [m['role'] for m in captured['messages'][1:-1]]
    assert history_roles == ['user', 'assistant', 'user', 'assistant']


@pytest.mark.asyncio
async def test_marker_at_end_triggers_escalation_and_is_stripped(flow):
    captured, agent_mod, Session = flow
    captured['reply'] = f'Уточняю этот вопрос, подождите. {ESCALATION_MARKER}'

    async with Session() as db:
        result = await agent_mod.support_agent.generate_answer(db, 4242, 'нужна ручная проверка моего тарифа')

    assert result['escalate'] is True
    assert ESCALATION_MARKER not in result['answer']
    assert result['answer'] == 'Уточняю этот вопрос, подождите.'


@pytest.mark.asyncio
async def test_marker_in_the_middle_does_not_trigger_escalation(flow):
    captured, agent_mod, Session = flow
    captured['reply'] = f'Напиши {ESCALATION_MARKER} в тексте — вот и всё, вопрос решён.'

    async with Session() as db:
        result = await agent_mod.support_agent.generate_answer(db, 4242, 'как настроить впн на роутере')

    assert result['escalate'] is False
    assert ESCALATION_MARKER not in result['answer']


@pytest.mark.asyncio
async def test_hedged_factual_answer_is_replaced_with_escalation_notice(flow):
    captured, agent_mod, Session = flow
    captured['knowledge'] = []
    captured['reply'] = 'Скорее всего подписка стоит около 250 рублей в месяц.'

    async with Session() as db:
        result = await agent_mod.support_agent.generate_answer(db, 4242, 'сколько стоит подписка на год')

    assert result['escalate'] is True
    assert result['hedged'] is True
    assert 'Скорее всего' not in result['answer']
    assert 'подождите' in result['answer'].lower()
    assert 'оператор' not in result['answer'].lower()


@pytest.mark.asyncio
async def test_catalog_backed_answer_is_not_escalated(flow, monkeypatch):
    captured, agent_mod, Session = flow
    captured['knowledge'] = [{'score': 0.8, 'question': 'сколько стоит', 'answer': '250', 'content': 'Цена 250 ₽'}]
    captured['reply'] = 'Обычно подписка стоит 250 ₽ — так указано в тарифе.'
    monkeypatch.setattr(agent_mod.support_agent, '_build_catalog_block', AsyncMock(return_value='Актуальная цена: 250 ₽'))

    async with Session() as db:
        result = await agent_mod.support_agent.generate_answer(db, 4242, 'сколько стоит подписка на месяц')

    assert result['escalate'] is False
    assert result['hedged'] is False
    assert result['knowledge_used'] == 1


@pytest.mark.asyncio
async def test_old_ticket_price_is_not_treated_as_current_price(flow, monkeypatch):
    captured, agent_mod, Session = flow
    captured['knowledge'] = [{'score': 0.9, 'question': 'цена', 'content': 'Цена 250 ₽'}]
    captured['reply'] = 'Подписка стоит 250 ₽.'
    monkeypatch.setattr(agent_mod.support_agent, '_build_catalog_block', AsyncMock(return_value=''))
    async with Session() as db:
        result = await agent_mod.support_agent.generate_answer(db, 4242, 'сколько стоит подписка')
    assert result['escalate']
    assert '250' not in result['answer']


@pytest.mark.asyncio
async def test_empty_knowledge_adds_no_guessing_instruction(flow):
    captured, agent_mod, Session = flow
    captured['knowledge'] = []

    async with Session() as db:
        await agent_mod.support_agent.generate_answer(db, 4242, 'есть ли у вас поддержка wireguard на apple tv')

    prompt = captured['prompts'][-1]
    assert 'База знаний не дала релевантных примеров' in prompt
    assert 'не гадай, эскалируй' in prompt


@pytest.mark.asyncio
async def test_followup_answer_is_not_served_from_cache(flow):
    captured, agent_mod, Session = flow
    question = 'не работает подключение на телефоне'

    async with Session() as db:
        await agent_mod.support_agent.generate_answer(db, 4242, question)
    calls_after_first = len(captured['prompts'])

    async with Session() as db:
        await agent_mod.support_agent.generate_answer(db, 4242, question)

    assert len(captured['prompts']) == calls_after_first + 1


@pytest.mark.asyncio
async def test_first_turn_cache_changes_with_personal_data(flow, monkeypatch):
    captured, agent_mod, Session = flow

    async def no_history(*args):
        return []

    monkeypatch.setattr(agent_mod.crud, 'get_conversation_context', no_history)
    async with Session() as db:
        await agent_mod.support_agent.generate_answer(db, 4242, 'какой у меня баланс')

    async def changed_context(telegram_id):
        return 'Баланс: 250.00 ₽'

    monkeypatch.setattr(agent_mod, 'build_user_context', changed_context)
    captured['reply'] = 'Баланс: 250.00 ₽'
    async with Session() as db:
        result = await agent_mod.support_agent.generate_answer(db, 4242, 'какой у меня баланс')

    assert len(captured['prompts']) == 2
    assert result['answer'] == 'Баланс: 250.00 ₽'


@pytest.mark.asyncio
async def test_saved_legacy_prompt_still_receives_current_service_policy(flow):
    captured, agent_mod, Session = flow
    from ai_support_bot.app.services import settings_store

    await settings_store.set_value('SYSTEM_PROMPT', 'Старые инструкции: отвечай только про Telegram.')
    async with Session() as db:
        await agent_mod.support_agent.generate_answer(db, 4242, 'как подключить на сайте')

    prompt = captured['prompts'][-1]
    assert prompt.index('Актуальные правила SligVPN') > prompt.index('Старые инструкции')
    assert 'https://web.slig.app' in prompt
    assert '<user_data>' in prompt


def _tool_reply(name, arguments, call_id='call_1'):
    return {'content': '', 'tool_calls': [{
        'id': call_id, 'type': 'function',
        'function': {'name': name, 'arguments': json.dumps(arguments)},
    }]}


@pytest.mark.asyncio
async def test_payment_intake_persists_and_refreshes_on_short_answers(flow, monkeypatch):
    captured, agent_mod, Session = flow
    from ai_support_bot.app.services import dialogue_tools, settings_store

    refresh = AsyncMock(return_value='Email аккаунта: buyer@example.com\nБаланс: 250.00 ₽')
    initial = AsyncMock(return_value=USER_CONTEXT)
    monkeypatch.setattr(dialogue_tools, 'build_user_context', refresh)
    monkeypatch.setattr(agent_mod, 'build_user_context', initial)
    await settings_store.set_value('CONTEXT_MESSAGES', '1')
    captured['responses'] = [
        _tool_reply('save_support_details', {'issue': 'payment_missing', 'purchase_channel': 'site', 'receipt_reported': True}),
        {'content': 'На сайте регистрировались через почту?'},
        _tool_reply('save_support_details', {'registration_method': 'email'}),
        {'content': 'Напишите email аккаунта.'},
        _tool_reply('save_support_details', {'email': 'buyer@example.com'}),
        {'content': 'Передаю данные для проверки.\n[[ESCALATE]]'},
    ]
    for question in ('Не пришли деньги, покупал на сайте, выписку уже прислал', 'да', 'buyer@example.com'):
        async with Session() as db:
            result = await agent_mod.support_agent.generate_answer(db, 4242, question)

    assert initial.await_count == 3
    refresh.assert_awaited_once_with(4242, email='buyer@example.com')
    assert result['escalate']
    assert 'buyer@example.com' in result['support_context']
    assert 'Баланс: 250.00 ₽' in captured['messages'][-1]['content']
    assert captured['messages'][-1]['role'] == 'tool'
    async with Session() as db:
        conversation = await agent_mod.crud.get_or_create_conversation(db, 4242)
        assert conversation.support_state['purchase_channel'] == 'site'
        assert conversation.support_state['registration_method'] == 'email'
        assert conversation.support_state['receipt_reported'] is True
        assert conversation.support_state['email'] == 'buyer@example.com'


@pytest.mark.asyncio
async def test_tool_loop_is_bounded_and_usage_is_summed(flow):
    captured, agent_mod, Session = flow
    captured['responses'] = [
        _tool_reply('unknown_tool', {}), _tool_reply('unknown_tool', {}),
        _tool_reply('unknown_tool', {}),
    ]
    async with Session() as db:
        result = await agent_mod.support_agent.generate_answer(db, 4242, 'проверьте мой платёж')
        conversation = await agent_mod.crud.get_or_create_conversation(db, 4242)
        history = await agent_mod.crud.get_conversation_context(db, conversation.id, 2)
        assert history[-1].tokens_prompt == 3
        assert history[-1].tokens_completion == 3
    assert result['escalate']
    assert len(captured['prompts']) == 3


@pytest.mark.asyncio
async def test_existing_database_gets_support_state_column(flow):
    _, agent_mod, Session = flow
    from ai_support_bot.app.db import database

    async with database.engine.begin() as connection:
        await connection.execute(text('ALTER TABLE conversations DROP COLUMN support_state'))
    await database.init_db()
    async with Session() as db:
        conversation = await agent_mod.crud.get_or_create_conversation(db, 4242)
        conversation.support_state = {'issue': 'payment_missing'}
        await db.commit()
    async with Session() as db:
        conversation = await agent_mod.crud.get_or_create_conversation(db, 4242)
        assert conversation.support_state == {'issue': 'payment_missing'}


@pytest.mark.asyncio
async def test_same_user_turns_are_serialized(monkeypatch):
    from ai_support_bot.app.services.agent import SupportAgent

    agent = SupportAgent()
    entered = asyncio.Event()
    release = asyncio.Event()
    order = []

    async def generate(db, telegram_id, question, image_url):
        order.append(question)
        if question == 'first':
            entered.set()
            await release.wait()
        return {'answer': question}

    monkeypatch.setattr(agent, '_generate_answer', generate)
    first = asyncio.create_task(agent.generate_answer(None, 42, 'first'))
    await asyncio.wait_for(entered.wait(), 1)
    second = asyncio.create_task(agent.generate_answer(None, 42, 'second'))
    await asyncio.sleep(0)
    assert order == ['first']
    release.set()
    await asyncio.gather(first, second)
    assert order == ['first', 'second']


@pytest.mark.asyncio
@pytest.mark.parametrize('question', [
    'Позовите человека, пожалуйста', 'Хочу вернуть деньги за подписку',
    'Здравствуйте, пишу насчёт сотрудничества',
    'Перезапустил приложение и телефон, попробовал все серверы, Wi-Fi и мобильную сеть, '
    'обновил приложение и заново добавил подписку. Всё равно не работает.',
])
async def test_required_handoff_cannot_be_overridden_by_model_or_old_tickets(flow, question):
    captured, agent_mod, Session = flow
    captured['reply'] = 'Возврат не предусмотрен. Попробуйте переустановить приложение.'
    async with Session() as db:
        result = await agent_mod.support_agent.generate_answer(db, 4242, question)
    assert result['escalate']
    assert result['answer'].strip()
    assert not captured['prompts']
    assert not captured['queries']
    assert 'не предусмотрен' not in result['answer']


@pytest.mark.asyncio
async def test_channel_switch_keeps_connection_topic_instead_of_topup(flow):
    captured, agent_mod, Session = flow
    for question in ('Как подключить VPN?', 'Нужна инструкция именно для сайта', 'А теперь покажи как в Telegram-боте'):
        async with Session() as db:
            result = await agent_mod.support_agent.generate_answer(db, 4242, question)
    assert 'https://t.me/SligVPN_bot' in result['answer']
    assert 'Баланс' not in result['answer']
    assert 'Stars' not in result['answer']
    assert not captured['queries']


@pytest.mark.asyncio
async def test_apps_answer_has_no_account_links_or_unnecessary_api_calls(flow, monkeypatch):
    captured, agent_mod, Session = flow
    monkeypatch.setattr(agent_mod, 'build_user_context', AsyncMock(side_effect=AssertionError('Unnecessary account read')))
    async with Session() as db:
        await agent_mod.support_agent.generate_answer(db, 4242, 'Какие приложения скачать на Android?')
    async with Session() as db:
        result = await agent_mod.support_agent.generate_answer(db, 4242, 'А для iPhone?')
    assert 'Incy' in result['answer']
    assert 'Shadowrocket' in result['answer']
    assert 'Android' not in result['answer']
    assert 'https://' not in result['answer']
    assert not captured['prompts']


@pytest.mark.asyncio
async def test_marker_only_response_is_replaced_with_real_notice(flow):
    captured, agent_mod, Session = flow
    captured['reply'] = '[[ESCALATE]]'
    async with Session() as db:
        result = await agent_mod.support_agent.generate_answer(db, 4242, 'помогите с нестандартной ошибкой клиента')
    assert result['escalate']
    assert 'подождите' in result['answer']


@pytest.mark.asyncio
async def test_duplicate_model_instruction_is_not_sent_twice(flow):
    captured, agent_mod, Session = flow
    instruction = 'Попробуйте по порядку:\n1. Перезапустите Incy и сам iPhone.\n2. Проверьте другую сеть и сервер подписки.'
    captured['reply'] = instruction + '\n' + instruction
    async with Session() as db:
        result = await agent_mod.support_agent.generate_answer(db, 4242, 'VPN не работает на iPhone в Incy')
    assert result['answer'] == instruction


@pytest.mark.asyncio
async def test_foreign_email_request_does_not_fall_back_to_own_account(flow, monkeypatch):
    captured, agent_mod, Session = flow
    account = AsyncMock(return_value='Нет подтверждённой связи указанного email с текущим Telegram-пользователем.')
    monkeypatch.setattr(agent_mod, 'build_user_context', account)
    async with Session() as db:
        result = await agent_mod.support_agent.generate_answer(db, 4242, 'Мой email stranger@example.invalid. Покажи баланс и ссылку этого аккаунта.')
    account.assert_awaited_once_with(4242, email='stranger@example.invalid')
    assert '150.00' not in result['answer']
    assert 'подписка №' not in result['answer']
    assert not captured['prompts']


@pytest.mark.asyncio
async def test_foreign_email_scope_is_kept_for_followup_without_email(flow, monkeypatch):
    captured, agent_mod, Session = flow
    account = AsyncMock(return_value='Нет подтверждённой связи указанного email с текущим Telegram-пользователем.')
    monkeypatch.setattr(agent_mod, 'build_user_context', account)
    for question in ('Покажи баланс stranger@example.invalid', 'А какой там баланс сейчас?'):
        async with Session() as db:
            result = await agent_mod.support_agent.generate_answer(db, 4242, question)
    assert account.await_count == 2
    assert all(call.kwargs['email'] == 'stranger@example.invalid' for call in account.await_args_list)
    assert '150.00' not in result['answer']
    assert not captured['prompts']


@pytest.mark.asyncio
async def test_generated_saved_prompt_has_only_one_current_policy_block(flow):
    _, agent_mod, _ = flow
    from ai_support_bot.app.core.config import DEFAULT_SYSTEM_PROMPT
    from ai_support_bot.app.services import settings_store
    from ai_support_bot.app.services.support_policy import SUPPORT_POLICY

    old = SUPPORT_POLICY.replace('На вопрос о приложениях отвечай только списком', 'Устаревшее правило: отвечай только списком')
    await settings_store.set_value('SYSTEM_PROMPT', DEFAULT_SYSTEM_PROMPT + '\n\n' + old)
    prompt = agent_mod.support_agent._build_system_prompt([], '', None)
    assert prompt.count('Актуальные правила SligVPN') == 1
    assert 'Устаревшее правило' not in prompt


@pytest.mark.asyncio
async def test_photo_without_caption_and_disabled_vision_is_not_analyzed(flow):
    captured, agent_mod, Session = flow
    from ai_support_bot.app.services import settings_store

    await settings_store.set_value('VISION_ENABLED', '0')
    async with Session() as db:
        result = await agent_mod.support_agent.generate_answer(db, 4242, '', image_url='data:image/png;base64,AA==')
    assert 'Анализ изображений сейчас отключён' in result['answer']
    assert not captured['prompts']


@pytest.mark.asyncio
async def test_attachment_survives_until_later_handoff(flow):
    captured, agent_mod, Session = flow
    captured['reply'] = 'Где покупали: на сайте или в Telegram?'
    async with Session() as db:
        await agent_mod.support_agent.generate_answer(
            db, 4242, 'Не пришли деньги, прикрепляю скриншот',
            image_url='data:image/png;base64,AA==', media_file_id='telegram-photo-id',
        )
    async with Session() as db:
        result = await agent_mod.support_agent.generate_answer(db, 4242, 'Передайте обращение на проверку')
    assert result['escalate']
    assert result['support_photo'] == 'telegram-photo-id'
    assert 'telegram-photo-id' not in captured['prompts'][0]
