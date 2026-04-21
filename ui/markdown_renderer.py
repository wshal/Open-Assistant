"""
Custom markdown renderer for rich display in QTextEdit.
RESTORATION: High-performance regex-based syntax highlighting + premium styling.
"""

import re
import html
from typing import List, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)


# Color palette for syntax highlighting in code blocks (Midnight Theme)
SYNTAX_COLORS = {
    "keyword": "#c586c0",  # purple
    "string": "#ce9178",  # orange
    "comment": "#6a9955",  # green
    "number": "#b5cea8",  # light green
    "function": "#dcdcaa",  # yellow
    "type": "#4ec9b0",  # teal
    "operator": "#d4d4d4",  # white
    "decorator": "#d7ba7d",  # gold
    "builtin": "#569cd6",  # blue
}

# Keywords by language for basic syntax highlighting
KEYWORDS = {
    "python": [
        "def", "class", "import", "from", "return", "if", "elif", "else",
        "for", "while", "try", "except", "finally", "with", "as", "yield",
        "async", "await", "lambda", "pass", "break", "continue", "raise",
        "True", "False", "None", "and", "or", "not", "in", "is", "self",
    ],
    "javascript": [
        "function", "const", "let", "var", "return", "if", "else", "for",
        "while", "switch", "case", "break", "continue", "new", "this",
        "class", "extends", "import", "export", "default", "async", "await",
        "try", "catch", "throw", "typeof", "instanceof", "true", "false",
        "null", "undefined", "yield", "of", "in",
    ],
    "generic": [
        "if", "else", "for", "while", "return", "class", "function", "def",
        "import", "from", "try", "catch", "throw", "new", "true", "false",
        "null", "void", "int", "string", "bool", "float", "double", "var",
        "let", "const", "public", "private", "protected", "static",
    ],
}


class MarkdownRenderer:
    """
    Convert markdown text to styled HTML for QTextEdit display.
    """

    def __init__(self, font_family: str = "Segoe UI", code_font: str = "Cascadia Code"):
        self.font_family = font_family
        self.code_font = f"'{code_font}', 'Fira Code', 'Consolas', monospace"
        self._in_code_block = False
        self._code_lang = ""
        self._code_buffer = []

    def render(self, text: str) -> str:
        """Convert markdown to styled HTML."""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return ""

        # Pre-clean: Remove any corrupted HTML style attributes
        text = re.sub(r'style="color:\s*#[^"]*";?', "", text)

        lines = text.split("\n")
        html_parts = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Code blocks (```)
            if line.strip().startswith("```"):
                if self._in_code_block:
                    # End code block
                    code_content = "\n".join(self._code_buffer)
                    html_parts.append(
                        self._render_code_block(code_content, self._code_lang)
                    )
                    self._in_code_block = False
                    self._code_buffer = []
                    self._code_lang = ""
                else:
                    # Start code block
                    self._in_code_block = True
                    self._code_lang = line.strip()[3:].strip().lower()
                    self._code_buffer = []
                i += 1
                continue

            if self._in_code_block:
                self._code_buffer.append(line)
                i += 1
                continue

            # Process non-code lines
            rendered = self._render_line(line)
            if rendered is not None:
                html_parts.append(rendered)

            i += 1

        # Close unclosed code block
        if self._in_code_block and self._code_buffer:
            code_content = "\n".join(self._code_buffer)
            html_parts.append(self._render_code_block(code_content, self._code_lang))
            self._in_code_block = False

        body = "\n".join(html_parts)

        return f"""
        <div style="font-family: {self.font_family}, sans-serif;
                    font-size: 13px; line-height: 1.6; color: #d0d0e8;">
            {body}
        </div>
        """

    def _render_line(self, line: str) -> str:
        """Render a single markdown line to HTML."""
        if line is None:
            return ""
        stripped = line.strip()

        if not stripped:
            return '<div style="height: 8px;"></div>'

        # Horizontal rule
        if re.match(r"^(-{3,}|\*{3,}|_{3,})\s*$", stripped):
            return '<hr style="border: none; border-top: 1px solid rgba(100,100,160,30); margin: 12px 0;">'

        # Headers
        h_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if h_match:
            level = len(h_match.group(1))
            text = self._render_inline(h_match.group(2))
            sizes = {1: 20, 2: 17, 3: 15, 4: 14, 5: 13, 6: 12}
            size = sizes.get(level, 13)
            color = "#c8c8ff" if level <= 2 else "#a0a0dd"
            margin = "16px 0 8px 0" if level <= 2 else "12px 0 6px 0"
            border = (
                "border-bottom: 1px solid rgba(100,100,160,20); padding-bottom: 4px;"
                if level <= 2
                else ""
            )
            return f'<div style="font-size: {size}px; font-weight: bold; color: {color}; margin: {margin}; {border}">{text}</div>'

        # Task lists
        task_match = re.match(r"^[-*]\s+\[([ xX])\]\s+(.+)$", stripped)
        if task_match:
            checked = task_match.group(1).lower() == "x"
            text = self._render_inline(task_match.group(2))
            icon = "[x]" if checked else "[ ]"
            strike = "text-decoration: line-through; opacity: 0.6;" if checked else ""
            return f'<div style="margin: 3px 0; padding-left: 4px; {strike}">{icon} {text}</div>'

        # Bullet lists
        ul_match = re.match(r"^(\s*)([-*+])\s+(.+)$", stripped)
        if ul_match:
            indent = len(ul_match.group(1)) // 2
            text = self._render_inline(ul_match.group(3))
            margin_left = 12 + indent * 16
            return f'<div style="margin: 3px 0; padding-left: {margin_left}px;">• {text}</div>'

        # Ordered lists
        ol_match = re.match(r"^(\s*)(\d+)[.\)]\s+(.+)$", stripped)
        if ol_match:
            indent = len(ol_match.group(1)) // 2
            num = ol_match.group(2)
            text = self._render_inline(ol_match.group(3))
            margin_left = 12 + indent * 16
            return f'<div style="margin: 3px 0; padding-left: {margin_left}px;">{num}. {text}</div>'

        # Blockquote
        if stripped.startswith(">"):
            text = self._render_inline(stripped.lstrip("> "))
            return (
                f'<div style="margin: 6px 0; padding: 6px 12px; '
                f"border-left: 3px solid rgba(100,100,200,60); "
                f"background: rgba(40,40,80,40); border-radius: 0 4px 4px 0; "
                f'color: #9999cc; font-style: italic;">{text}</div>'
            )

        # Table row detection
        if "|" in stripped and stripped.startswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if all(re.match(r"^[-: ]+$", c) for c in cells):
                return ""
            cell_html = "".join(
                f'<td style="padding: 4px 10px; border: 1px solid rgba(60,60,100,30);">'
                f"{self._render_inline(c)}</td>"
                for c in cells
            )
            return f'<tr style="background: rgba(30,30,50,80);">{cell_html}</tr>'

        # Regular paragraph
        text = self._render_inline(stripped)
        return f'<div style="margin: 3px 0;">{text}</div>'

    def _render_inline(self, text: str) -> str:
        """Render inline markdown elements."""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return ""

        # Escape HTML first
        text = html.escape(text)

        # Bold + Italic
        text = re.sub(
            r"\*\*\*(.+?)\*\*\*|___(.+?)___",
            lambda m: f"<b><i>{m.group(1) or m.group(2)}</i></b>",
            text,
        )

        # Bold
        text = re.sub(
            r"\*\*(.+?)\*\*|__(.+?)__",
            lambda m: f'<b style="color: #e0e0ff;">{m.group(1) or m.group(2)}</b>',
            text,
        )

        # Italic
        text = re.sub(
            r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)",
            lambda m: f'<i style="color: #b8b8dd;">{m.group(1) or m.group(2)}</i>',
            text,
        )

        # Strikethrough
        text = re.sub(r"~~(.+?)~~", r'<s style="opacity: 0.6;">\1</s>', text)

        # Inline code
        text = re.sub(
            r"`([^`]+)`",
            lambda m: (
                f'<code style="background: rgba(40,40,70,150); color: #e6b673; '
                f'padding: 1px 6px; border-radius: 3px; font-size: 12px; '
                f'font-family: {self.code_font};">{m.group(1)}</code>'
            ),
            text,
        )

        # Links
        text = re.sub(
            r"\[(.+?)\]\((.+?)\)",
            r'<a style="color: #6699cc; text-decoration: underline;" href="\2">\1</a>',
            text,
        )

        return text

    def _render_code_block(self, code: str, language: str = "") -> str:
        """Render a fenced code block with syntax highlighting."""
        highlighted = self._syntax_highlight(code, language)

        # Language label is a SIBLING div placed above the code block — NOT nested
        # inside it. QTextEdit's HTML engine leaks text-align from nested children
        # into the parent block, which was right-aligning all the code content.
        # Keeping them as siblings gives each its own independent text-align scope.
        lang_label = ""
        if language:
            lang_label = (
                f'<div style="text-align: right; color: #7777aa; font-size: 10px; '
                f'font-family: {self.font_family}, sans-serif; '
                f'margin: 8px 0 0 0; padding: 0 2px;">'
                f"{language}</div>"
            )

        code_block = (
            f'<div style="background: rgba(8,8,18,220); border: 1px solid rgba(80,80,140,60); '
            f"border-radius: 8px; margin: 2px 0 12px 0; padding: 14px; "
            f"font-family: {self.code_font}; font-size: 12px; "
            f'line-height: 1.5; white-space: pre; text-align: left;">'
            f"{highlighted}"
            f"</div>"
        )

        return lang_label + code_block


    def _apply_highlight(self, chunks, regex, style: str):
        highlighted = []
        for chunk in chunks:
            if chunk["type"] != "raw":
                highlighted.append(chunk)
                continue

            text = chunk["text"]
            last_index = 0
            for match in regex.finditer(text):
                if match.start() > last_index:
                    highlighted.append({"type": "raw", "text": text[last_index:match.start()]})
                highlighted.append({"type": "styled", "style": style, "text": match.group(0)})
                last_index = match.end()

            highlighted.append({"type": "raw", "text": text[last_index:]})

        return highlighted

    def _syntax_highlight(self, code: str, language: str) -> str:
        """Basic syntax highlighting for code."""
        if not code:
            return ""

        code = html.escape(str(code))
        chunks = [{"type": "raw", "text": code}]

        string_pattern = re.compile(r'(&quot;.*?&quot;|&#x27;.*?&#x27;|"[^"]*"|\'[^\']*\')', re.DOTALL)
        comment_pattern = re.compile(
            r'(?:(?<!&)(?:#(?![0-9A-Za-z]+;)[^\n]*|//[^\n]*|/\*[\s\S]*?\*/))',
            re.MULTILINE,
        )
        number_pattern = re.compile(r"\b(\d+(?:\.\d*)?)\b")

        lang_keywords = KEYWORDS.get(language, KEYWORDS["generic"])
        keyword_pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(kw) for kw in sorted(lang_keywords, key=len, reverse=True)) + r")\b"
        )

        chunks = self._apply_highlight(chunks, string_pattern, "string")
        chunks = self._apply_highlight(chunks, comment_pattern, "comment")
        chunks = self._apply_highlight(chunks, number_pattern, "number")
        chunks = self._apply_highlight(chunks, keyword_pattern, "keyword")

        result = []
        for chunk in chunks:
            if chunk["type"] == "raw":
                result.append(chunk["text"])
            else:
                color = SYNTAX_COLORS.get(chunk["style"], SYNTAX_COLORS["operator"])
                result.append(
                    f'<span style="color: {color};">{chunk["text"]}</span>'
                )

        return "".join(result)
