import re
from decimal import Decimal


NOTICE = 'Передаю обращение команде для проверки, подождите, пожалуйста.'
CLIENTS = {
    'Android': ('Incy', 'Happ', 'V2RayNG'),
    'iPhone/iPad': ('Incy', 'Happ', 'Shadowrocket', 'Streisand'),
    'Windows': ('Incy', 'Happ', 'v2rayN'),
    'macOS': ('Incy', 'Happ', 'V2Box'),
}


def smalltalk_answer(question: str) -> str | None:
    text = question.strip().casefold().rstrip('!. 🙂😊👍')
    if text in {'привет', 'здравствуйте', 'доброе утро', 'добрый день', 'добрый вечер'}:
        return 'Здравствуйте!'
    if text in {'спасибо', 'спасибо большое', 'благодарю', 'спс'}:
        return 'Пожалуйста!'
    if text in {'ок', 'окей', 'понял', 'ясно', 'хорошо'}:
        return 'Хорошо.'
    return None


def mandatory_handoff(question: str, history: list) -> str | None:
    text = question.casefold()
    if re.fullmatch(r'\s*(?:человек|оператор|админ)[!.?\s]*', text):
        return 'human'
    human = re.search(r'(?:позов\w*|зови|свяж\w*|соедин\w*|подключ\w*|хочу поговорить|нужен|нужна)\s+(?:меня\s+)?(?:с\s+)?(?:жив\w*\s+)?(?:человек\w*|оператор\w*|админ\w*|поддержк\w*)', text)
    if human and not re.search(r'не\s*$', text[:human.start()]):
        return 'human'
    if re.search(r'передайте\s+(?:это\s+)?обращение|передайте\s+.*команде', text):
        return 'human'
    if re.search(r'(?:хочу|прошу|можно|нужно)\s+(?:сделать\s+)?(?:возврат|вернуть деньги)|верните\s+(?:мне\s+)?деньги|возврат средств', text):
        return 'refund'
    if re.search(r'сотрудничеств|предложени\w*\s+(?:о\s+)?реклам', text) and not re.search(r'условия реферальн|как.*реферальн', text):
        return 'partnership'
    if re.search(r'списал\w*.*(?:продлен|продлён|продление)', text) and re.search(r'(?:срок|подписк).*не (?:измен|продл)', text):
        return 'payment_dispute'
    previous = ' '.join(getattr(item, 'content', '') or '' for item in history[-4:]).casefold()
    technical = bool(re.search(r'vpn|впн|сервер|приложени|incy|happ|подключени', text + ' ' + previous))
    failed = bool(re.search(r'не помог\w*|вс[её] равно не работает|не заработал', text))
    actions = sum(bool(re.search(pattern, text)) for pattern in (
        r'перезапуст|перезагруз', r'сервер', r'wi.?fi|мобильн', r'обновил', r'заново|повторн\w* импорт',
    ))
    if technical and failed and (actions >= 2 or re.search(r'вс[её]\s+(?:это\s+)?(?:уже\s+)?(?:пробовал|сделал)', text)):
        return 'diagnostics_failed'
    return None


def handoff_answer(reason: str, question: str) -> str:
    if reason == 'partnership':
        return 'Спасибо за предложение о сотрудничестве! Передаю его команде, пожалуйста, подождите.'
    if not re.search(r'[а-яё]', question, re.IGNORECASE):
        return 'I am passing your request to the support team for review. Please wait.'
    return NOTICE


def apps_answer(question: str, history: list) -> str | None:
    text = question.casefold()
    previous = ' '.join(getattr(item, 'content', '') or '' for item in history[-4:]).casefold()
    missing = 'happ' in text and bool(re.search(r'нет|не (?:найти|вижу|отображ|доступ)', text)) and 'app store' in text
    app_request = bool(re.search(r'каки\w* приложени|како\w* (?:приложени|клиент)|что скачать|приложени\w* скачать', text))
    followup = bool(re.match(r'^а (?:для|на)\b', text)) and bool(re.search(r'каки\w* приложени|incy|shadowrocket|v2rayng', previous))
    if not (app_request or followup or missing):
        return None
    if missing:
        return 'Если Happ не отображается в App Store, используйте Incy, Shadowrocket или Streisand. Доступность приложения зависит от региона магазина.'
    patterns = {'Android': r'android|андроид', 'iPhone/iPad': r'iphone|ipad|ios|айфон', 'Windows': r'windows|виндов', 'macOS': r'macos|макбук|mac\b'}
    platforms = [platform for platform, pattern in patterns.items() if re.search(pattern, text)] or list(CLIENTS)
    return '\n'.join(f'{platform}: {", ".join(CLIENTS[platform])}.' for platform in platforms)


def is_payment_missing(question: str) -> bool:
    return bool(re.search(
        r'(?:деньги|плат[её]ж|пополнени\w*|оплата)\s+не\s+(?:приш|поступ|зачис|появ)|не\s+(?:приш|поступ|зачис|появ)\w*\s+(?:деньги|плат[её]ж|оплата)|баланс\s+не\s+попол|не\s+пополни\w*\s+баланс',
        question, re.IGNORECASE,
    ))


def compact_knowledge(knowledge: list[dict], max_chars: int = 2400) -> list[dict]:
    result = []
    remaining = max_chars
    for item in knowledge:
        content = (item.get('content') or '')[:min(1200, remaining)]
        if not content or remaining < 100:
            break
        content = re.sub(r'(?:https?|vless|vmess|trojan|ss|happ|incy)://[^\s<>"\x27]+', '[ссылка из прошлого обращения скрыта]', content)
        content = re.sub(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', '[email из прошлого обращения скрыт]', content, flags=re.IGNORECASE)
        content = content[:remaining]
        result.append({**item, 'content': content})
        remaining -= len(content)
    return result


def deduplicate_answer(answer: str) -> str:
    answer = answer.strip()
    for _ in range(3):
        match = re.fullmatch(r'(.{80,}?)\s*\1', answer, re.DOTALL)
        if not match:
            break
        answer = match[1].strip()
    return answer


def unsupported_price(answer: str, question: str, catalog: str) -> bool:
    if not re.search(r'стоит|стоимост|цен[ауы]|поч[её]м', question, re.IGNORECASE):
        return False

    def amounts(text):
        values = re.findall(r'((?:\d{1,3}(?:[ \u00a0]\d{3})+|\d+)(?:[.,]\d{1,2})?)\s*(?:₽|руб\w*)', text, re.IGNORECASE)
        return {Decimal(''.join(value.split()).replace(',', '.')) for value in values}

    return bool(amounts(answer) - amounts(catalog))
