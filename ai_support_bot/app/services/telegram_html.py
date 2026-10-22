import re
from html import escape
from html.parser import HTMLParser
from urllib.parse import urlsplit


def _bot_deep_link(url: str) -> str:
    if re.fullmatch(r'https://t\.me/SligVPN_bot/?', url, flags=re.IGNORECASE):
        return 'https://t.me/SligVPN_bot?start=ai_support'
    return url


def _link_in_text(match: re.Match) -> str:
    original = match.group()
    url = original.rstrip('.,;!):]')
    return _bot_deep_link(url) + original[len(url):]


class _TelegramHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        tag = {'strong': 'b', 'em': 'i'}.get(tag, tag)
        if tag in {'b', 'i', 'code'}:
            self.parts.append(f'<{tag}>')
            self.stack.append(tag)
        elif tag == 'a':
            href = _bot_deep_link(dict(attrs).get('href') or '')
            try:
                parsed = urlsplit(href)
                valid = parsed.scheme == 'https' and parsed.hostname and not parsed.username and not any(ord(c) < 32 for c in href)
            except ValueError:
                valid = False
            if valid:
                self.parts.append(f'<a href="{escape(href, quote=True)}">')
                self.stack.append('a')
        elif tag == 'br':
            self.parts.append('\n')

    def handle_endtag(self, tag):
        tag = {'strong': 'b', 'em': 'i'}.get(tag, tag)
        if tag in self.stack:
            while self.stack:
                current = self.stack.pop()
                self.parts.append(f'</{current}>')
                if current == tag:
                    break

    def handle_data(self, data):
        data = re.sub(r'https://[^\s<>\"\']+', _link_in_text, data, flags=re.IGNORECASE)
        self.parts.append(escape(data, quote=False))


def sanitize_telegram_html(text: str) -> str:
    parser = _TelegramHTML()
    parser.feed(text)
    parser.close()
    parser.parts.extend(f'</{tag}>' for tag in reversed(parser.stack))
    rendered = ''.join(parser.parts).strip()
    return re.sub(r'<code>\s*(https://[^\s<>]+)\s*</code>', r'\1', rendered, flags=re.IGNORECASE)
