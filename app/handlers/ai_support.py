import html

import structlog
from aiogram import Dispatcher, F, types
from aiogram.fsm.context import FSMContext
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database.models import User
from app.localization.texts import get_texts
from app.services.ai.ai_assistant_service import AIAssistantService
from app.states import AIAssistantStates
from app.utils.cache import RateLimitCache


logger = structlog.get_logger(__name__)


def _intro_text(texts, company: str) -> str:
    return texts.t(
        'AI_ASSISTANT_INTRO',
        '🤖 <b>ИИ-помощник {company}</b>\n\n'
        'Опишите ваш вопрос обычным сообщением — я постараюсь помочь сразу: '
        'настройка, подписка, оплата, устройства и т.д.\n\n'
        'Если понадобится живой оператор — просто напишите «оператор», и я передам обращение.',
    ).format(company=company)


def _chat_keyboard(texts) -> types.InlineKeyboardMarkup:
    return types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text=texts.t('AI_ASSISTANT_CALL_OPERATOR', '🆘 Позвать оператора'),
                    callback_data='ai_call_operator',
                )
            ],
            [
                types.InlineKeyboardButton(
                    text=texts.t('AI_ASSISTANT_NEW_TOPIC', '🔄 Новый вопрос'), callback_data='ai_reset'
                )
            ],
            [
                types.InlineKeyboardButton(
                    text=texts.t('BACK_TO_MENU', '🏠 В главное меню'), callback_data='back_to_menu'
                )
            ],
        ]
    )


async def start_ai_assistant(callback: types.CallbackQuery, state: FSMContext, db_user: User):
    if not AIAssistantService.is_available():
        await callback.answer(
            get_texts(db_user.language).t('AI_ASSISTANT_DISABLED', 'ИИ-помощник сейчас недоступен.'),
            show_alert=True,
        )
        return

    texts = get_texts(db_user.language)
    company = settings.AI_ASSISTANT_COMPANY_NAME or 'поддержки'
    await state.set_state(AIAssistantStates.chatting)

    try:
        await callback.message.edit_text(
            _intro_text(texts, company), reply_markup=_chat_keyboard(texts), parse_mode='HTML'
        )
    except Exception:
        await callback.message.answer(
            _intro_text(texts, company), reply_markup=_chat_keyboard(texts), parse_mode='HTML'
        )
    await callback.answer()


async def _create_escalation_ticket(db: AsyncSession, db_user: User, last_message: str) -> None:
    from app.database.crud.ticket import TicketCRUD
    from app.handlers.tickets import notify_admins_about_new_ticket

    try:
        if await TicketCRUD.user_has_active_ticket(db, db_user.id):
            return
        blocked_until = await TicketCRUD.is_user_globally_blocked(db, db_user.id)
        if blocked_until:
            return

        title = 'Обращение от ИИ-помощника'
        body = (last_message or '').strip()[:500] or 'Клиент запросил оператора через ИИ-помощника.'
        ticket = await TicketCRUD.create_ticket(db, db_user.id, title, body, 'normal')
        await db.commit()
        await notify_admins_about_new_ticket(ticket, db)
    except Exception as error:
        logger.error('Failed to escalate AI conversation to ticket', error=str(error), user_id=db_user.id)


async def handle_ai_message(message: types.Message, state: FSMContext, db_user: User, db: AsyncSession):
    current_state = await state.get_state()
    if current_state != AIAssistantStates.chatting:
        return

    texts = get_texts(db_user.language)

    interval = max(0, settings.AI_ASSISTANT_MIN_INTERVAL_SECONDS)
    if interval:
        limited = await RateLimitCache.is_rate_limited(db_user.id, 'ai_assistant_msg', limit=1, window=interval)
        if limited:
            return

    text = (message.text or message.caption or '').strip()
    if not text:
        await message.answer(
            texts.t('AI_ASSISTANT_TEXT_ONLY', 'Пожалуйста, опишите вопрос текстом 🙂'),
            reply_markup=_chat_keyboard(texts),
        )
        return

    handoff_keyword = (settings.AI_ASSISTANT_HANDOFF_KEYWORD or '').strip().lower()
    force_handoff = bool(handoff_keyword) and handoff_keyword in text.lower()

    try:
        await message.bot.send_chat_action(message.chat.id, 'typing')
    except Exception:
        pass

    reply = await AIAssistantService.process_message(db, db_user, text)
    await db.commit()

    if reply.error:
        await message.answer(
            texts.t(
                'AI_ASSISTANT_ERROR',
                '😔 Не удалось получить ответ. Передаю ваш вопрос оператору — он скоро ответит.',
            ),
            reply_markup=_chat_keyboard(texts),
        )
        await _create_escalation_ticket(db, db_user, text)
        return

    if reply.limited:
        await message.answer(
            texts.t(
                'AI_ASSISTANT_LIMIT',
                'На сегодня лимит сообщений ИИ-помощнику исчерпан. Создайте тикет — ответит оператор.',
            ),
            reply_markup=_chat_keyboard(texts),
        )
        return

    answer = reply.text or texts.t('AI_ASSISTANT_EMPTY', 'Уточните, пожалуйста, ваш вопрос 🙂')
    await message.answer(html.escape(answer), reply_markup=_chat_keyboard(texts))

    if reply.handoff or force_handoff:
        await _create_escalation_ticket(db, db_user, text)
        await message.answer(
            texts.t(
                'AI_ASSISTANT_HANDED_OFF',
                '✅ Передал ваш вопрос живому оператору — он ответит в тикете в ближайшее время.',
            )
        )


async def call_operator(callback: types.CallbackQuery, state: FSMContext, db_user: User, db: AsyncSession):
    texts = get_texts(db_user.language)
    await _create_escalation_ticket(db, db_user, 'Клиент запросил оператора через ИИ-помощника.')
    await state.clear()
    try:
        await callback.message.edit_text(
            texts.t(
                'AI_ASSISTANT_HANDED_OFF',
                '✅ Передал ваш вопрос живому оператору — он ответит в тикете в ближайшее время.',
            ),
            reply_markup=types.InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        types.InlineKeyboardButton(
                            text=texts.t('BACK_TO_MENU', '🏠 В главное меню'), callback_data='back_to_menu'
                        )
                    ]
                ]
            ),
        )
    except Exception:
        pass
    await callback.answer()


async def reset_ai_conversation(callback: types.CallbackQuery, state: FSMContext, db_user: User, db: AsyncSession):
    texts = get_texts(db_user.language)
    await AIAssistantService.reset(db, db_user.id)
    await db.commit()
    await state.set_state(AIAssistantStates.chatting)
    company = settings.AI_ASSISTANT_COMPANY_NAME or 'поддержки'
    try:
        await callback.message.edit_text(
            _intro_text(texts, company), reply_markup=_chat_keyboard(texts), parse_mode='HTML'
        )
    except Exception:
        pass
    await callback.answer(texts.t('AI_ASSISTANT_RESET_DONE', 'Начинаем новый вопрос 🔄'))


def register_handlers(dp: Dispatcher):
    dp.callback_query.register(start_ai_assistant, F.data == 'ai_assistant')
    dp.callback_query.register(call_operator, F.data == 'ai_call_operator')
    dp.callback_query.register(reset_ai_conversation, F.data == 'ai_reset')
    dp.message.register(handle_ai_message, AIAssistantStates.chatting)
