import json
import re

from ai_support_bot.app.services.user_data import build_user_context


_EMAIL_RE = re.compile(r'[A-Z0-9.!#$%&\x27*+/=?^_`{|}~-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)+', re.IGNORECASE)
_ENUMS = {
    'issue': {'payment_missing', 'other', 'resolved'},
    'purchase_channel': {'site', 'telegram', 'unknown'},
    'registration_method': {'email', 'telegram', 'other', 'unknown'},
}

SUPPORT_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'save_support_details',
            'description': 'Сохрани новые сведения пользователя. При уточнении канала, способа регистрации или email автоматически обновляет данные доступного аккаунта. Не заполняй неизвестное догадками. Исправления заменяют прежнее; новый несвязанный вопрос отмечай issue=other.',
            'parameters': {
                'type': 'object',
                'properties': {
                    **{key: {'type': 'string', 'enum': sorted(values)} for key, values in _ENUMS.items()},
                    'email': {'type': 'string', 'description': 'Email аккаунта, явно сообщённый пользователем; пустая строка удаляет ошибочный email.'},
                    'receipt_reported': {'type': 'boolean', 'description': 'Пользователь сообщил, что прислал чек или выписку; это не подтверждение платежа.'},
                    'new_case': {'type': 'boolean', 'description': 'Начался новый отдельный платёжный случай: очисти сведения прежнего. Не ставь true для продолжения текущего обращения.'},
                },
                'additionalProperties': False,
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'refresh_account_data',
            'description': 'Заново прочитай актуальные данные аккаунта после уточнений или сообщения об оплате. Для отдельного email-аккаунта укажи сообщённый email. Доступ разрешён только при связи с текущим Telegram ID; чужие аккаунты не доступны.',
            'parameters': {
                'type': 'object',
                'properties': {'email': {'type': 'string'}},
                'additionalProperties': False,
            },
        },
    },
]


def support_state_block(state: dict) -> str:
    if not state:
        return ''
    return (
        '<support_details>\nСведения со слов пользователя, не подтверждённые факты об аккаунте. '
        'Используй для текущей темы; не возвращайся к старому вопросу при смене темы.\n'
        + json.dumps(state, ensure_ascii=False)
        + '\n</support_details>'
    )


def _valid_email(value: object, evidence: str, state: dict) -> bool:
    if not isinstance(value, str) or len(value) > 254 or not _EMAIL_RE.fullmatch(value):
        return False
    emails = {item.casefold() for item in _EMAIL_RE.findall(evidence)}
    if state.get('email'):
        emails.add(state['email'].casefold())
    return value.casefold() in emails


async def execute_support_tool(name: str, raw_arguments: str, telegram_id: int, state: dict, evidence: str) -> dict:
    try:
        if len(raw_arguments) > 2000:
            return {'error': 'Аргументы слишком длинные.'}
        args = json.loads(raw_arguments)
    except (ValueError, TypeError):
        return {'error': 'Некорректный JSON аргументов.'}
    if not isinstance(args, dict):
        return {'error': 'Ожидался объект аргументов.'}
    if name == 'save_support_details':
        allowed = {*_ENUMS, 'email', 'receipt_reported', 'new_case'}
        if args.keys() - allowed:
            return {'error': 'Неизвестные поля.'}
        for key, values in _ENUMS.items():
            if key in args and (not isinstance(args[key], str) or args[key] not in values):
                return {'error': f'Некорректное значение {key}.'}
        if 'email' in args and args['email'] != '' and not _valid_email(args['email'], evidence, state):
            return {'error': 'Email должен быть явно сообщён пользователем.'}
        for field in ('receipt_reported', 'new_case'):
            if field in args and not isinstance(args[field], bool):
                return {'error': f'Некорректное значение {field}.'}
        if args.pop('new_case', False) or args.get('issue') in {'other', 'resolved'}:
            state.clear()
        if 'purchase_channel' in args and state.get('purchase_channel') != args['purchase_channel']:
            state.pop('email', None)
            state.pop('registration_method', None)
        changed = any(state.get(key) != value for key, value in args.items())
        state.update(args)
        if state.get('email') == '':
            state.pop('email', None)
        output = {'saved': True, 'reported_details': dict(state)}
        if changed and args.keys() & {'purchase_channel', 'registration_method', 'email'}:
            output['refresh'] = await execute_support_tool('refresh_account_data', '{}', telegram_id, state, evidence)
        return output
    if name == 'refresh_account_data':
        if args.keys() - {'email'}:
            return {'error': 'Неизвестные поля.'}
        email = args.get('email')
        if email is not None and not _valid_email(email, evidence, state):
            return {'error': 'Email должен быть явно сообщён пользователем.'}
        if not email and state.get('purchase_channel') == 'site':
            email = state.get('email')
            if state.get('registration_method') != 'telegram' and not email:
                return {'account_data': 'Аккаунт покупки на сайте пока не определён. Уточни способ регистрации, а при входе через почту — email. Данные Telegram нельзя считать данными этого аккаунта.'}
        context = await build_user_context(telegram_id, email=email) if email else await build_user_context(telegram_id)
        return {'account_data': context, 'scope': 'Только аккаунт, связанный с текущим Telegram ID.'}
    return {'error': 'Неизвестный инструмент.'}


def support_details_for_admin(state: dict) -> str:
    if not state or state.get('issue') in {'other', 'resolved'}:
        return ''
    labels = {
        'purchase_channel': 'Место покупки', 'registration_method': 'Способ регистрации',
        'email': 'Указанный email', 'receipt_reported': 'Сообщил о чеке/выписке',
    }
    values = {'site': 'сайт', 'telegram': 'Telegram', 'email': 'почта', 'other': 'другое', 'unknown': 'неизвестно'}
    lines = ['Сведения со слов пользователя (владение email и платёж не подтверждены):']
    for key, label in labels.items():
        if key in state:
            raw = state[key]
            value = ('да' if raw else 'нет') if isinstance(raw, bool) else values.get(raw, raw)
            lines.append(f'{label}: {value}')
    return '\n'.join(lines)
