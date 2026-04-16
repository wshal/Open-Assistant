"""
Full-session history feed view (Single Page).
Displays all conversation data in a scrollable list.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel, QFrame, QApplication, QPushButton
)
from PyQt6.QtCore import Qt, QTimer
from ui.markdown_renderer import MarkdownRenderer

class HistoryFeedView(QWidget):
    def __init__(self, history, parent=None):
        super().__init__(parent)
        self.history = history
        self.md = MarkdownRenderer()
        self._setup_ui()

    def _setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # Header with optional Back button
        self.header = QFrame()
        self.hl = QHBoxLayout(self.header)
        self.hl.setContentsMargins(0, 0, 0, 0)
        
        self.back_btn = QPushButton("← Sessions")
        self.back_btn.setStyleSheet("""
            QPushButton { background: #3a3a5e; color: #ccc; border-top-left-radius: 6px; border-bottom-left-radius: 6px; padding: 4px 10px; font-size: 11px; }
            QPushButton:hover { background: #4a4a7e; color: white; }
        """)
        self.back_btn.clicked.connect(self.show_sessions_list)
        self.back_btn.hide()
        self.hl.addWidget(self.back_btn)
        
        self.title_label = QLabel("📚 History Library")
        self.title_label.setStyleSheet("color: #a0a0cc; font-weight: bold; font-size: 13px;")
        self.hl.addWidget(self.title_label)
        self.hl.addStretch()
        
        self.main_layout.addWidget(self.header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(15)
        self.content_layout.addStretch()

        self.scroll.setWidget(self.content_widget)
        self.main_layout.addWidget(self.scroll)

    def refresh(self):
        """Called when opening the feed - default to sessions list."""
        self.show_sessions_list()

    def show_sessions_list(self):
        self._clear_layout()
        self.back_btn.hide()
        self.title_label.setText("📚 Sessions Library")
        
        sessions = self.history.sessions
        if not sessions:
            lbl = QLabel("No history sessions found.")
            lbl.setStyleSheet("color: #556; font-style: italic; margin-top: 20px;")
            self.content_layout.insertWidget(0, lbl)
            return

        import datetime
        for s in sessions:
            card = QFrame()
            card.setStyleSheet("""
                QFrame { background: rgba(30,30,50,180); border-radius: 8px; border: 1px solid rgba(80,80,150,30); }
                QFrame:hover { border: 1px solid #558; background: rgba(40,40,70,220); }
            """)
            cl = QVBoxLayout(card)
            
            ts = datetime.datetime.fromtimestamp(s['created_at']).strftime("%Y-%m-%d %H:%M")
            header = QHBoxLayout()
            date_lbl = QLabel(ts)
            date_lbl.setStyleSheet("color: #778; font-size: 10px;")
            header.addWidget(date_lbl)
            header.addStretch()
            count = QLabel(f"💬 {s.get('entry_count', 0)}")
            count.setStyleSheet("color: #668; font-size: 10px;")
            header.addWidget(count)
            cl.addLayout(header)
            
            snip = QLabel(s.get('snippet', '...'))
            snip.setStyleSheet("color: #ccc; font-size: 11px; margin-top: 5px;")
            snip.setWordWrap(True)
            cl.addWidget(snip)
            
            # Make card clickable
            card.mousePressEvent = lambda e, sid=s['id']: self.show_session_detail(sid)
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            
            self.content_layout.insertWidget(self.content_layout.count() - 1, card)

    def show_session_detail(self, session_id):
        self._clear_layout()
        self.back_btn.show()
        self.title_label.setText("💬 Session Detail")
        
        self.history.load_session(session_id)
        
        for entry in self.history.entries:
            # Q
            if entry.query:
                q = QLabel(f"Q: {entry.query}")
                q.setWordWrap(True)
                q.setStyleSheet("background: rgba(80,80,160,30); color: #a0a0ff; padding: 8px; border-radius: 6px; font-size: 11px;")
                self.content_layout.insertWidget(self.content_layout.count() - 1, q)
            
            # A
            a_frame = QFrame()
            a_frame.setStyleSheet("background: rgba(30,30,50,220); color: #d0d0e8; padding: 10px; border-radius: 8px;")
            al = QVBoxLayout(a_frame)
            ans = QLabel(self.md.render(entry.response))
            ans.setWordWrap(True)
            ans.setTextFormat(Qt.TextFormat.RichText)
            ans.setOpenExternalLinks(True)
            al.addWidget(ans)
            self.content_layout.insertWidget(self.content_layout.count() - 1, a_frame)
        
        QTimer.singleShot(50, lambda: self.scroll.verticalScrollBar().setValue(0))

    def _clear_layout(self):
        while self.content_layout.count() > 1:
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
