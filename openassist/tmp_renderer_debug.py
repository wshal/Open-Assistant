import re
import html
from ui.markdown_renderer import SYNTAX_COLORS, KEYWORDS

code = '''// Example of Props
function Greeting(props) {
  return <h1>Hello, {props.name}!</h1>;
}'''

escaped = html.escape(code)
print('ESCAPED:')
print(escaped)

# Strings
string_pattern = r'(&quot;.*?&quot;|&#x27;.*?&#x27;|"[^"]*"|\'[^\']*\'')'
repl = f'<span style="color: {SYNTAX_COLORS["string"]};">\\1</span>'
step1 = re.sub(string_pattern, repl, escaped)
print('\nAFTER STRINGS:')
print(step1)

# Numbers
step2 = re.sub(r"\b(\d+\.?\d*)\b", f'<span style="color: {SYNTAX_COLORS["number"]};">\\1</span>', step1)
print('\nAFTER NUMBERS:')
print(step2)

# Keywords
lang_keywords = KEYWORDS.get('javascript', KEYWORDS['generic'])
step3 = step2
for kw in sorted(lang_keywords, key=len, reverse=True):
    step3 = re.sub(rf"\b({re.escape(kw)})\b", f'<span style="color: {SYNTAX_COLORS["keyword"]};">\\1</span>', step3)
print('\nAFTER KEYWORDS:')
print(step3)

# Comments
step4 = re.sub(r"(#[^\n]*|//[^\n]*)", f'<span style="color: {SYNTAX_COLORS["comment"]};">\\1</span>', step3)
print('\nAFTER COMMENTS:')
print(step4)
