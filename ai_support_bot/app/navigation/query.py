import re


def navigation_target(query: str) -> str | None:
    text = query.casefold()
    if re.search(r'колес|вращен|выигранн', text):
        return 'web_wheel_history'
    if re.search(r'промокод|промик', text):
        return 'promocode'
    if re.search(r'привяз\w*.*(?:telegram|телеграм|аккаунт)|привязанн\w* аккаунт', text):
        return 'web_accounts'
    if re.search(r'смен\w*.*(?:email|почт)|измен\w*.*email', text):
        return 'web_change_email'
    if re.search(r'автоплат|автопродл', text):
        return 'subscription_autopay'
    if re.search(r'продл', text):
        return 'subscription_extend'
    if re.search(r'куп\w*.*подпис|оформ\w*.*подпис', text):
        return 'subs_buy_subscription'
    if re.search(r'пополн\w*.*баланс', text):
        return 'balance_topup'
    if re.search(r'истори\w*.*(?:операц|платеж|платёж)', text):
        return 'balance_history'
    if re.search(r'подключ\w*.*vpn|подключ\w*.*впн|ссылк\w*.*подключ|настро\w*.*vpn', text):
        return 'subscription_connect'
    return None


def contextual_navigation_query(question: str, history: list, summary: str | None = None) -> str:
    if navigation_target(question):
        return question.strip()
    followup = bool(re.search(r'^(?:а\b|а теперь|нужна инструкция|теперь|покажи|как там|где это)', question.strip(), re.IGNORECASE))
    if not followup and len(question.strip()) >= 25:
        return question.strip()
    for item in reversed(history):
        if getattr(item, 'role', '') != 'user':
            continue
        content = getattr(item, 'content', '') or ''
        if navigation_target(content):
            return content[:300] + '\n' + question.strip()
    if summary and followup:
        return summary[:300] + '\n' + question.strip()
    return question.strip()
