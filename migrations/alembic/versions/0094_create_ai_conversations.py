"""create ai_conversations table

Revision ID: 0094
Revises: 0093
Create Date: 2026-08-01

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0094'
down_revision: Union[str, None] = '0093'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if _table_exists('ai_conversations'):
        return

    op.create_table(
        'ai_conversations',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            'user_id',
            sa.Integer(),
            sa.ForeignKey('users.id', ondelete='CASCADE'),
            nullable=False,
        ),
        sa.Column('summary', sa.Text(), nullable=False, server_default=''),
        sa.Column('recent_messages', postgresql.JSONB(), nullable=False, server_default='[]'),
        sa.Column('user_message_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_messages', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('messages_today', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('day_bucket', sa.String(10), nullable=True),
        sa.Column('handed_off', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('last_message_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('user_id', name='uq_ai_conversations_user'),
    )
    op.create_index('ix_ai_conversations_user_id', 'ai_conversations', ['user_id'])


def downgrade() -> None:
    op.drop_index('ix_ai_conversations_user_id', table_name='ai_conversations')
    op.drop_table('ai_conversations')


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    result = bind.execute(
        sa.text('SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = :name)'),
        {'name': table_name},
    )
    return result.scalar()
