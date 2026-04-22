path = r'c:\Users\Vishal\Desktop\Open Assist\ui\overlay.py'

with open(path, 'r', encoding='utf-8') as f:
    body = f.read()

header = (
    '"""\n'
    'Main overlay window - v4.1 (Layer 4 Hardened).\n'
    'RESTORED: Markdown Render Debounce (150ms) and Manual Scroll Lock.\n'
    'FIXED: Connection of transcript and audio-status bridges.\n'
    'P0.1 FIX: Removed duplicate CRLF class definition artifact.\n'
    '"""\n'
    '\n'
    'import time\n'
    'from PyQt6.QtWidgets import (\n'
    '    QMainWindow,\n'
    '    QWidget,\n'
    '    QVBoxLayout,\n'
    '    QHBoxLayout,\n'
    '    QTextEdit,\n'
    '    QLineEdit,\n'
    '    QPushButton,\n'
    '    QLabel,\n'
    '    QFrame,\n'
    '    QApplication,\n'
    '    QStackedWidget,\n'
    ')\n'
    'from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPoint\n'
    'from PyQt6.QtGui import QTextCursor, QTextCharFormat, QColor\n'
)

with open(path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(header + body)

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
print(f'Total lines: {len(lines)}')
for i, l in enumerate(lines[:12]):
    print(f'{i+1}: {repr(l.rstrip())}')
