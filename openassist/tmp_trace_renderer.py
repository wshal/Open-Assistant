from ui.markdown_renderer import MarkdownRenderer

text = '''```javascript
// Example of Props
function Greeting(props) {
  return <h1>Hello, {props.name}!</h1>;
}
```'''

md = MarkdownRenderer()

code = text.split('```', 2)[1].split('\n', 1)[1]
print('RAW CODE:\n', code)
escaped = __import__('html').escape(code)
print('ESCAPED:\n', escaped)
print('--- string replacement ---')
import re
from ui.markdown_renderer import SYNTAX_COLORS
string_pattern = r'(&quot;.*?&quot;|&#x27;.*?&#x27;|"[^"]*"|\'[^\']*\'')'
repl = f'<span style="color: {SYNTAX_COLORS["string"]};">\\1</span>'
step = re.sub(string_pattern, repl, escaped)
print(step)
print('--- keywords ---')
print(step)
