"""Theme engine with multiple built-in themes."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Theme:
    name: str
    bg_primary: str
    bg_secondary: str
    bg_input: str
    text_primary: str
    text_secondary: str
    text_muted: str
    accent: str
    accent_hover: str
    border: str
    success: str
    error: str
    warning: str


THEMES: Dict[str, Theme] = {
    "midnight": Theme(
        name="Midnight",
        bg_primary="rgba(16,16,28,250)",
        bg_secondary="rgba(12,12,22,230)",
        bg_input="rgba(24,24,44,220)",
        text_primary="#d0d0e8",
        text_secondary="#a0a0cc",
        text_muted="#555577",
        accent="rgba(70,70,170,200)",
        accent_hover="rgba(90,90,190,220)",
        border="rgba(70,70,120,50)",
        success="#44aa44",
        error="#cc4444",
        warning="#ccaa44",
    ),
    "dark": Theme(
        name="Dark",
        bg_primary="rgba(30,30,30,250)",
        bg_secondary="rgba(20,20,20,230)",
        bg_input="rgba(45,45,45,220)",
        text_primary="#e0e0e0",
        text_secondary="#a0a0a0",
        text_muted="#666666",
        accent="rgba(60,120,200,200)",
        accent_hover="rgba(80,140,220,220)",
        border="rgba(80,80,80,50)",
        success="#44bb44",
        error="#dd4444",
        warning="#ddaa44",
    ),
    "nord": Theme(
        name="Nord",
        bg_primary="rgba(46,52,64,250)",
        bg_secondary="rgba(36,42,54,230)",
        bg_input="rgba(59,66,82,220)",
        text_primary="#eceff4",
        text_secondary="#d8dee9",
        text_muted="#4c566a",
        accent="rgba(136,192,208,200)",
        accent_hover="rgba(129,161,193,220)",
        border="rgba(76,86,106,50)",
        success="#a3be8c",
        error="#bf616a",
        warning="#ebcb8b",
    ),
    "dracula": Theme(
        name="Dracula",
        bg_primary="rgba(40,42,54,250)",
        bg_secondary="rgba(30,32,44,230)",
        bg_input="rgba(68,71,90,220)",
        text_primary="#f8f8f2",
        text_secondary="#bd93f9",
        text_muted="#6272a4",
        accent="rgba(189,147,249,200)",
        accent_hover="rgba(255,121,198,220)",
        border="rgba(68,71,90,80)",
        success="#50fa7b",
        error="#ff5555",
        warning="#f1fa8c",
    ),
    "solarized": Theme(
        name="Solarized Dark",
        bg_primary="rgba(0,43,54,250)",
        bg_secondary="rgba(7,54,66,230)",
        bg_input="rgba(0,43,54,220)",
        text_primary="#839496",
        text_secondary="#93a1a1",
        text_muted="#586e75",
        accent="rgba(38,139,210,200)",
        accent_hover="rgba(42,161,152,220)",
        border="rgba(88,110,117,50)",
        success="#859900",
        error="#dc322f",
        warning="#b58900",
    ),
    "light": Theme(
        name="Light",
        bg_primary="rgba(248,248,252,250)",
        bg_secondary="rgba(240,240,245,230)",
        bg_input="rgba(255,255,255,220)",
        text_primary="#2d2d3d",
        text_secondary="#5d5d7d",
        text_muted="#9d9dbd",
        accent="rgba(60,100,180,200)",
        accent_hover="rgba(80,120,200,220)",
        border="rgba(200,200,220,80)",
        success="#228822",
        error="#cc2222",
        warning="#aa8800",
    ),
}


class ThemeEngine:
    """Apply themes to the application."""

    def __init__(self, config):
        self.config = config
        theme_name = config.get("app.theme", "midnight")
        self.current = THEMES.get(theme_name, THEMES["midnight"])

    def set_theme(self, name: str):
        if name in THEMES:
            self.current = THEMES[name]
            self.config.set("app.theme", name)

    def get_overlay_stylesheet(self) -> str:
        t = self.current
        return f"""
            #box {{
                background: {t.bg_primary};
                border: 1px solid {t.border};
                border-radius: 14px;
            }}
            QTextEdit {{
                background: {t.bg_secondary};
                color: {t.text_primary};
                border: 1px solid {t.border};
                border-radius: 10px;
                padding: 12px;
                font-size: 13px;
                font-family: 'Cascadia Code', 'Fira Code', monospace;
            }}
            QLineEdit {{
                background: {t.bg_input};
                color: {t.text_primary};
                border: 1px solid {t.border};
                border-radius: 20px;
                padding: 9px 18px;
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border-color: {t.accent};
            }}
            QPushButton#sendButton {{
                background: {t.accent};
                color: white;
                border: none;
                border-radius: 18px;
                font-weight: bold;
            }}
            QPushButton#sendButton:hover {{
                background: {t.accent_hover};
            }}
            QLabel {{
                color: {t.text_secondary};
            }}
            QComboBox {{
                background: {t.bg_input};
                color: {t.text_secondary};
                border: 1px solid {t.border};
                border-radius: 6px;
                padding: 3px 8px;
            }}
        """

    @staticmethod
    def available_themes() -> list:
        return list(THEMES.keys())