from datetime import UTC, datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database.models import User


logger = structlog.get_logger(__name__)


def _display_name(user: User) -> str:
    parts = [p for p in [user.first_name, user.last_name] if p]
    if parts:
        return ' '.join(parts)
    if user.username:
        return f'@{user.username}'
    return 'клиент'


async def build_user_context(db: AsyncSession, user: User) -> str:
    """Собирает компактную сводку о пользователе для контекста ИИ.

    Цель — чтобы ассистент максимально знал клиента (баланс, подписка, статус,
    язык, реферальная история), но при этом не раздувал промпт. Только факты,
    коротко, одной строкой на пункт.
    """

    lines: list[str] = []
    lines.append(f'Имя: {_display_name(user)}')
    if user.username:
        lines.append(f'Username: @{user.username}')
    lines.append(f'ID клиента: {user.id}')
    if user.language:
        lines.append(f'Язык: {user.language}')

    try:
        lines.append(f'Баланс: {settings.format_price(user.balance_kopeks or 0)}')
    except Exception:
        lines.append(f'Баланс (коп.): {user.balance_kopeks or 0}')

    lines.append('Делал платную подписку ранее: ' + ('да' if user.has_had_paid_subscription else 'нет'))
    lines.append('Пополнял баланс: ' + ('да' if getattr(user, 'has_made_first_topup', False) else 'нет'))

    if user.created_at:
        try:
            days = (datetime.now(UTC) - user.created_at).days
            lines.append(f'В сервисе дней: {max(days, 0)}')
        except Exception:
            pass

    if user.referral_code:
        lines.append(f'Реферальный код: {user.referral_code}')
    if user.referred_by_id:
        lines.append('Пришёл по приглашению другого пользователя')

    try:
        from app.database.crud.subscription import get_subscription_by_user_id

        subscription = await get_subscription_by_user_id(db, user.id)
    except Exception as error:
        logger.debug('Failed to load subscription for AI context', error=str(error))
        subscription = None

    if subscription:
        status = getattr(subscription, 'status', None)
        lines.append(f'Подписка: есть (статус: {status})')
        if getattr(subscription, 'is_trial', False):
            lines.append('Тип подписки: пробная')
        end_date = getattr(subscription, 'end_date', None)
        if end_date:
            try:
                days_left = (end_date - datetime.now(UTC)).days
                lines.append(f'Подписка активна до: {end_date.strftime("%d.%m.%Y")} (осталось дней: {days_left})')
            except Exception:
                pass
        limit_gb = getattr(subscription, 'traffic_limit_gb', None)
        used_gb = getattr(subscription, 'traffic_used_gb', None)
        if limit_gb == 0:
            lines.append('Трафик: безлимит')
        elif limit_gb:
            lines.append(f'Трафик: использовано {used_gb or 0} / {limit_gb} ГБ')
        device_limit = getattr(subscription, 'device_limit', None)
        if device_limit:
            lines.append(f'Лимит устройств: {device_limit}')
        lines.append(
            'Автопродление: ' + ('включено' if getattr(subscription, 'autopay_enabled', False) else 'выключено')
        )
    else:
        lines.append('Подписка: отсутствует (клиент ещё не покупал или закончилась)')

    return '\n'.join(lines)
