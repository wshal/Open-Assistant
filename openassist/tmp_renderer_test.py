from ui.markdown_renderer import MarkdownRenderer

text = '''```javascript
// Example of Props
function Greeting(props) {
  return <h1>Hello, {props.name}!</h1>;
}
```'''

md = MarkdownRenderer()
print(md.render(text))
