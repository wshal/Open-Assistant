from pathlib import Path
path = Path(r'c:\Users\Vishal\Desktop\New\openassist\ui\markdown_renderer.py')
text = path.read_text(encoding='utf-8')
old_block = '''        # Comments
        code = re.sub(
            r"(#[^\\n]*|//[^\\n]*)",
            f'<span style="color: {SYNTAX_COLORS["comment"]};">\\1</span>',
            code,
        )

        # Strings (Correctly handle html-escaped quotes)
'''
new_block = '        # Strings (Correctly handle html-escaped quotes)\n'
if old_block not in text:
    raise RuntimeError('original comment block not found')
text = text.replace(old_block, new_block, 1)
insert_block = '''        # Comments last so the generated HTML from earlier replacements is not reprocessed
        code = re.sub(
            r"(#[^\\n]*|//[^\\n]*)",
            f'<span style="color: {SYNTAX_COLORS["comment"]};">\\1</span>',
            code,
        )\n\n'''
if '        return code\n' not in text:
    raise RuntimeError('return block not found')
text = text.replace('        return code\n', insert_block + '        return code\n', 1)
path.write_text(text, encoding='utf-8')
print('patched')
