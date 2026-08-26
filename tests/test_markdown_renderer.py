import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.markdown_renderer import MarkdownRenderer


def test_markdown_table_is_wrapped_and_keeps_columns_horizontal():
    source = """| Decision | Custom&#x20;HOC | Higher‑Order Component (HOC) |
|---|---|---|
| Use case | Custom logic | Reusable component wrapper |
| Tradeoff | More code | Clear separation of concerns |
"""

    rendered = MarkdownRenderer().render(source)

    assert rendered.count("<table ") == 1
    assert rendered.count("<th ") == 3
    assert rendered.count("<td ") == 6
    assert "<tr" in rendered and "</table>" in rendered
    assert "Higher‑Order Component (HOC)" in rendered
    assert "Custom HOC" in rendered
    assert "&#x20;" not in rendered
    assert 'width="100%"' in rendered


def test_pipe_text_without_markdown_separator_is_not_treated_as_a_table():
    rendered = MarkdownRenderer().render("A | B")

    assert "<table " not in rendered
    assert "A | B" in rendered
