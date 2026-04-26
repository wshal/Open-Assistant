import os

def fix_file(path):
    with open(path, 'rb') as f:
        data = f.read()
    
    # Map of corrupted sequences to correct characters
    # These are common "Mojibake" patterns
    replacements = {
        b'\xc3\xa2\xe2\x82\xac\xc2\xa2': '•'.encode('utf-8'),
        b'\xc3\xa2\xca\x9c\xc2\x91': '☑'.encode('utf-8'),
        b'\xc3\xa2\xca\x9c\xc2\x90': '☐'.encode('utf-8'),
        b'\xc3\xa2\xc2\x9c\xc2\x85': '✅'.encode('utf-8'),
        b'\xc3\xa2\xc2\x9d\xc2\x8c': '❌'.encode('utf-8'),
        b'\xc3\xa2\xc2\x9a\xc2\xa0\xc3\xaf\xc2\xb8\xc2\x8f': '⚠️'.encode('utf-8'),
        b'\xc3\xb0\xc2\x9f\xc2\x92\xc2\xa1': '💡'.encode('utf-8'),
        b'\xc3\xb0\xc2\x9f\xc2\x94\xc2\xa5': '🔥'.encode('utf-8'),
        b'\xc3\xa2\xc2\xad\xc2\x90': '⭐'.encode('utf-8'),
        b'\xc3\xb0\xc2\x9f\xc2\x9a\xc2\x80': '🚀'.encode('utf-8'),
        b'\xc3\xb0\xc2\x9f\xc2\x9b\xc2\x9b': '🐛'.encode('utf-8'),
        b'\xc3\xb0\xc2\x9f\xc2\x91\xc2\x89': '👉'.encode('utf-8'),
    }

    new_data = data
    for old, new in replacements.items():
        new_data = new_data.replace(old, new)
    
    if new_data != data:
        with open(path, 'wb') as f:
            f.write(new_data)
        print(f"Fixed {path}")
    else:
        print(f"No changes needed for {path}")

    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    target_file = os.path.join(project_root, 'ui', 'markdown_renderer.py')
    fix_file(target_file)
