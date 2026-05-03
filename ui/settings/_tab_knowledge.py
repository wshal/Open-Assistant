from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea
)
from PyQt6.QtCore import Qt
from ui.settings.constants import TEXT_MUTED


class KnowledgeTabMixin:
    def _tab_knowledge(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(14, 16, 14, 16)
        l.setSpacing(20)
        w.setStyleSheet("background: transparent;")

        lbl_kb = self._make_section_label("KNOWLEDGE BASE")
        l.addWidget(lbl_kb)

        kb_desc = QLabel(
            "Feed the AI your own documents. Drop PDFs, Q&A JSON files, or plain text "
            "into the knowledge folder and they will be indexed automatically on next startup. "
            "Or use the button below to import files right now."
        )
        kb_desc.setWordWrap(True)
        kb_desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(kb_desc)

        self._kb_status_lbl = QLabel("Loading knowledge base info...")
        self._kb_status_lbl.setStyleSheet(
            "color: #818cf8; font-size: 10px; background: transparent;"
        )
        l.addWidget(self._kb_status_lbl)
        self._refresh_kb_status()

        kb_row = QHBoxLayout()
        kb_row.setSpacing(8)

        btn_import_kb = QPushButton("Import Files")
        btn_import_kb.setFixedHeight(30)
        btn_import_kb.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_import_kb.setToolTip(
            "Import PDF, JSON Q&A, or text files into the knowledge base.\n"
            "Files are copied to the knowledge folder and indexed immediately."
        )
        btn_import_kb.setStyleSheet("""
            QPushButton {
                background: rgba(99,102,241,35); color: #818cf8;
                border: 1px solid rgba(99,102,241,76); border-radius: 8px;
                font-size: 11px; font-weight: 700; padding: 0 12px;
            }
            QPushButton:hover { background: rgba(99,102,241,60); color: white; }
        """)
        btn_import_kb.clicked.connect(self._import_knowledge_files)
        kb_row.addWidget(btn_import_kb)

        btn_open_kb = QPushButton("Open Folder")
        btn_open_kb.setFixedHeight(30)
        btn_open_kb.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_open_kb.setToolTip("Open the knowledge documents folder in Explorer")
        btn_open_kb.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,8); color: #94a3b8;
                border: 1px solid rgba(255,255,255,15); border-radius: 8px;
                font-size: 11px; padding: 0 12px;
            }
            QPushButton:hover { background: rgba(255,255,255,18); color: white; }
        """)
        btn_open_kb.clicked.connect(self._open_knowledge_folder)
        kb_row.addWidget(btn_open_kb)
        kb_row.addStretch()
        l.addLayout(kb_row)

        l.addStretch()
        scroll.setWidget(w)
        return scroll
