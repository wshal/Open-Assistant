import sys

path = r'C:\Users\Vishal\Desktop\New\openassist\ui\markdown_renderer.py'
with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if '":bug:":' in line:
        new_lines.append('            ":bug:": "🐛",\n')
    else:
        new_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print(f"Fixed {path}")
