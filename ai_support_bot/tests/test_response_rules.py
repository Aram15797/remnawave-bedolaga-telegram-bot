from types import SimpleNamespace

import pytest

from ai_support_bot.app.navigation.query import contextual_navigation_query, navigation_target
from ai_support_bot.app.services.response_rules import compact_knowledge, is_payment_missing, mandatory_handoff
from ai_support_bot.app.services.telegram_html import sanitize_telegram_html


def test_long_channel_followup_uses_previous_user_topic():
    history = [
        SimpleNamespace(role='user', content='Как подключить VPN?'),
        SimpleNamespace(role='assistant', content='Баланс → Пополнить → Stars'),
        SimpleNamespace(role='user', content='Нужна инструкция именно для сайта'),
    ]
    query = contextual_navigation_query('А теперь покажи как в Telegram-боте', history)
    assert navigation_target(query) == 'subscription_connect'
    assert 'Stars' not in query


def test_context_budget_limits_seven_old_examples():
    knowledge = [{'content': 'x' * 1200, 'score': 0.9, 'question': str(i)} for i in range(7)]
    selected = compact_knowledge(knowledge)
    assert sum(len(item['content']) for item in selected) <= 2400
    assert len(selected) == 2


@pytest.mark.parametrize('question', ['Не подключайте оператора', 'Как продлить подписку?', 'VPN не работает', 'Какие условия реферальной программы?'])
def test_regular_questions_do_not_force_handoff(question):
    assert mandatory_handoff(question, []) is None


def test_telegram_html_closes_tags_and_escapes_text():
    assert sanitize_telegram_html('<b>Цена < 100 & 200') == '<b>Цена &lt; 100 &amp; 200</b>'


def test_telegram_html_preserves_safe_clickable_links():
    assert sanitize_telegram_html('<a href="https://web.slig.app/balance">Баланс</a>') == '<a href="https://web.slig.app/balance">Баланс</a>'


@pytest.mark.parametrize('url', ['javascript:alert(1)', 'https://user:password@example.com', 'file:///etc/passwd'])
def test_telegram_html_removes_unsafe_links(url):
    assert sanitize_telegram_html(f'<a href="{url}">Текст</a>') == 'Текст'


@pytest.mark.parametrize('question', ['Почему не пополнился баланс?', 'Не пришли деньги', 'Деньги не появились'])
def test_missing_payment_variants_start_same_workflow(question):
    assert is_payment_missing(question)


def test_topup_without_renewal_is_not_a_missing_payment():
    assert not is_payment_missing('Пополнил баланс, а подписка не продлилась. Почему?')


def test_past_ticket_private_links_and_emails_do_not_reach_model():
    knowledge = compact_knowledge([{'content': 'Адрес other@example.com, ссылка https://private.example/secret-token', 'score': 0.9}])
    assert 'other@example.com' not in knowledge[0]['content']
    assert 'secret-token' not in knowledge[0]['content']
    assert 'скрыт' in knowledge[0]['content']


@pytest.mark.parametrize(('state', 'expected'), [
    ({'receipt_reported': True}, 'Где вы оплачивали'),
    ({'purchase_channel': 'site', 'receipt_reported': True}, 'регистрировались'),
    ({'purchase_channel': 'site', 'registration_method': 'email'}, 'адрес электронной почты'),
    ({'purchase_channel': 'telegram'}, 'Пришлите чек'),
])
def test_payment_intake_asks_only_next_missing_detail(state, expected):
    from ai_support_bot.app.services.response_rules import payment_next_step

    answer, escalate = payment_next_step(state)
    assert expected in answer
    assert answer.count('?') <= 1
    assert not escalate


def test_complete_payment_intake_hands_off_without_repeating_receipt_request():
    from ai_support_bot.app.services.response_rules import payment_next_step

    answer, escalate = payment_next_step({'purchase_channel': 'site', 'registration_method': 'email', 'email': 'buyer@example.com', 'receipt_reported': True})
    assert escalate
    assert 'Передаю' in answer
    assert 'Пришлите' not in answer


def test_https_url_in_code_becomes_plain_clickable_url():
    assert sanitize_telegram_html('<code>https://web.slig.app/balance?a=1&amp;b=2</code>') == 'https://web.slig.app/balance?a=1&amp;b=2'
    assert sanitize_telegram_html('<code> HTTPS://web.slig.app </code>') == 'HTTPS://web.slig.app'
    assert sanitize_telegram_html('<code>some-token</code>') == '<code>some-token</code>'


@pytest.mark.parametrize('question', [
    'Деньги списали за продление, но подписка не продлилась',
    'Не пополнил баланс, подписка не продлилась',
    'Пополнил баланс, а подписка не продлилась. Когда сработает автоплатёж?',
])
def test_personal_payment_questions_are_not_generic_renewal_guidance(question):
    from ai_support_bot.app.services.response_rules import is_topup_without_renewal

    assert not is_topup_without_renewal(question)
