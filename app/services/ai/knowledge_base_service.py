import json
import re
from pathlib import Path

import structlog

from app.config import settings


logger = structlog.get_logger(__name__)

_WORD_RE = re.compile(r'[a-zA-Zа-яёА-ЯЁ0-9]{3,}')

_STOPWORDS = {
    'как',
    'что',
    'это',
    'для',
    'при',
    'или',
    'если',
    'меня',
    'мне',
    'вам',
    'нас',
    'the',
    'and',
    'for',
    'you',
    'your',
    'can',
    'not',
    'все',
    'уже',
    'так',
    'там',
    'мой',
    'моя',
    'мои',
    'вот',
    'над',
    'под',
    'без',
    'его',
    'ещё',
    'еще',
    'бы',
    'здравствуйте',
    'привет',
    'пожалуйста',
    'спасибо',
}


def _tokenize(text: str) -> set[str]:
    tokens = set()
    for match in _WORD_RE.findall((text or '').lower()):
        if match in _STOPWORDS:
            continue
        tokens.add(match)
        if len(match) > 4:
            tokens.add(match[:4])
    return tokens


class KnowledgeBaseService:
    """Retrieval поверх дистиллированной базы знаний.

    Хранит компактный набор FAQ-записей и по запросу подбирает наиболее
    релевантные, чтобы подмешать их в контекст LLM. Так модель отвечает по делу
    и не приходится отправлять всю базу целиком — это экономит токены.
    """

    _entries: list[dict] = []
    _loaded: bool = False

    @classmethod
    def _load(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True
        path = Path(settings.AI_ASSISTANT_KB_PATH)
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
            entries = raw.get('entries', []) if isinstance(raw, dict) else []
        except FileNotFoundError:
            logger.warning('AI knowledge base file not found', path=str(path))
            entries = []
        except Exception as error:
            logger.error('Failed to load AI knowledge base', error=str(error))
            entries = []

        prepared = []
        for entry in entries:
            keywords = entry.get('keywords', []) or []
            index_text = ' '.join(
                [
                    entry.get('topic', ''),
                    entry.get('q', ''),
                    ' '.join(keywords),
                ]
            )
            prepared.append(
                {
                    'id': entry.get('id', ''),
                    'topic': entry.get('topic', ''),
                    'q': entry.get('q', ''),
                    'a': entry.get('a', ''),
                    'keywords': [str(k).lower() for k in keywords],
                    'tokens': _tokenize(index_text),
                }
            )
        cls._entries = prepared

    @classmethod
    def reload(cls) -> None:
        cls._loaded = False
        cls._entries = []
        cls._load()

    @classmethod
    def is_available(cls) -> bool:
        cls._load()
        return bool(cls._entries)

    @classmethod
    def search(cls, query: str, top_k: int | None = None) -> list[dict]:
        cls._load()
        if not cls._entries:
            return []
        limit = top_k if top_k is not None else settings.AI_ASSISTANT_KB_TOP_K
        limit = max(1, int(limit or 1))

        query_lower = (query or '').lower()
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, dict]] = []
        for entry in cls._entries:
            score = 0.0
            for keyword in entry['keywords']:
                if keyword and keyword in query_lower:
                    score += 3.0
            overlap = len(query_tokens & entry['tokens'])
            score += overlap
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    @classmethod
    def build_context(cls, query: str, top_k: int | None = None) -> str:
        matches = cls.search(query, top_k=top_k)
        if not matches:
            return ''
        blocks = []
        for entry in matches:
            blocks.append(f'[{entry["topic"]}]\nВопрос: {entry["q"]}\nОтвет: {entry["a"]}')
        return '\n\n'.join(blocks)
