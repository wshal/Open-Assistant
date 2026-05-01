"""
ui/settings/constants.py
Shared styling and constants for the Settings UI.
"""

BG_DARK = "background: rgba(15,15,30,245);"
TEXT_PRIMARY = "color: #c0c0ff;"
TEXT_MUTED = "color: #64748b;"

SS_INPUT = """
QLineEdit, QComboBox {
    background: rgba(20,20,40,220);
    color: #e0e0f5;
    border: 1px solid rgba(80,85,255,20);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
}
"""

STYLE_BTN_PRIMARY = """
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4f46e5, stop:1 #7c3aed);
    color: white;
    border-radius: 12px;
    font-weight: 800;
    font-size: 12px;
    padding: 15px 30px;
    border: none;
}
QPushButton:hover {
    background: #6366f1;
    border: 1px solid rgba(255,255,255,20);
}
"""

STYLE_BTN_SECONDARY = """
QPushButton {
    background: rgba(255,255,255,12);
    color: #94a3b8;
    border-radius: 12px;
    font-weight: 600;
    font-size: 11px;
    padding: 12px 24px;
    border: 1px solid rgba(255,255,255,25);
}
QPushButton:hover {
    background: rgba(255,255,255,20);
    color: white;
}
"""
