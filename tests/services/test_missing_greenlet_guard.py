import pytest
from datetime import UTC, datetime
from types import SimpleNamespace

from app.database.models import Subscription, User
from app.services.subscription_service import SubscriptionService
from app.utils.subscription_utils import safe_get_attr
from tests.fixtures.sqlite_memory import memory_session


def test_safe_get_attr_primitives_and_dicts():
    assert safe_get_attr(None, 'id', default=42) == 42
    assert safe_get_attr(10, 'id', default=None) is None
    assert safe_get_attr('str', 'id', default=None) is None
    assert safe_get_attr({'id': 123, 'remnawave_id': 456}, 'id') == 123
    assert safe_get_attr({'id': 123, 'remnawave_id': 456}, 'remnawave_id') == 456
    assert safe_get_attr({'id': 123}, 'missing', default='fallback') == 'fallback'


def test_safe_get_attr_simplenamespace():
    ns = SimpleNamespace(id=99, remnawave_short_uuid='abc')
    assert safe_get_attr(ns, 'id') == 99
    assert safe_get_attr(ns, 'remnawave_short_uuid') == 'abc'
    assert safe_get_attr(ns, 'missing', default=None) is None


@pytest.mark.asyncio
async def test_safe_get_attr_expired_orm_instance(monkeypatch):
    """Accessing attributes on an expired ORM instance must NOT raise MissingGreenlet."""
    tables = [User.__table__, Subscription.__table__]
    async with memory_session(monkeypatch, tables) as db:
        user = User(id=1, telegram_id=123456)
        db.add(user)
        await db.commit()

        sub = Subscription(
            id=10,
            user_id=1,
            status='active',
            end_date=datetime(2030, 1, 1, tzinfo=UTC),
            remnawave_id=50,
            remnawave_short_uuid='test-uuid',
        )
        db.add(sub)
        await db.commit()

        # Force expire all instances (simulating post-rollback state)
        await db.rollback()

        # Primary key identity should return cleanly without loading
        assert safe_get_attr(sub, 'id') == 10

        # Expired attribute should return default without triggering _load_expired / MissingGreenlet
        assert safe_get_attr(sub, 'remnawave_short_uuid') is None or isinstance(
            safe_get_attr(sub, 'remnawave_short_uuid'), str
        )


@pytest.mark.asyncio
async def test_panel_id_is_free_for_with_expired_subscription(monkeypatch):
    """_panel_id_is_free_for must run cleanly on an expired Subscription ORM instance."""
    tables = [User.__table__, Subscription.__table__]
    async with memory_session(monkeypatch, tables) as db:
        user = User(id=1, telegram_id=123456)
        db.add(user)
        await db.commit()

        sub1 = Subscription(
            id=10,
            user_id=1,
            status='active',
            end_date=datetime(2030, 1, 1, tzinfo=UTC),
            remnawave_id=50,
        )
        db.add(sub1)
        await db.commit()

        # Expire all attributes (e.g. after rollback)
        await db.rollback()

        service = SubscriptionService()

        # Checking for panel_id=50 for sub1 (itself) should return True (free for sub1)
        assert await service._panel_id_is_free_for(db, sub1, 50) is True

        # Checking for panel_id=50 for a different sub_id (e.g. 20) should return False (taken by sub1)
        assert await service._panel_id_is_free_for(db, 20, 50) is False
