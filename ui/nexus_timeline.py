"""
P2.9 — Nexus Context Timeline View.

Shows a real-time scrollable log of all context events (audio, screen, RAG,
window-focus) collected by the Nexus engine during the current session.
Helps users understand what the AI "saw" before each response.
"""

import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Source → accent colour mapping
_SOURCE_COLOURS = {
    "audio":  "#4ade80",   # green
    "screen": "#60a5fa",   # blue
    "rag":    "#f59e0b",   # amber
    "window": "#a78bfa",   # violet
    "manual": "#94a3b8",   # grey
}

_SOURCE_ICONS = {
    "audio":  "🎙",
    "screen": "🖥",
    "rag":    "📚",
    "window": "🪟",
    "manual": "✏️",
}


class NexusTimelineView(QWidget):
    """Live context timeline panel — embedded in the main overlay stack."""

    def __init__(self, nexus, parent=None):
        super().__init__(parent)
        self.nexus = nexus
        self._last_entry_count = 0
        self._setup_ui()
        # Refresh every 2 seconds during a session
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh)
        self._refresh_timer.start(2000)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # ── Header ──────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("⏱ CONTEXT TIMELINE")
        title.setStyleSheet(
            "color: #a0a0cc; font-weight: 900; font-size: 11px; letter-spacing: 1.5px;"
        )
        hdr.addWidget(title)
        hdr.addStretch()

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedSize(52, 22)
        self.clear_btn.setStyleSheet(
            "background: rgba(239,68,68,0.12); color: #ef4444;"
            " border: 1px solid rgba(239,68,68,0.3); border-radius: 6px; font-size: 9px;"
            " QPushButton:hover { background: rgba(239,68,68,0.25); }"
        )
        self.clear_btn.clicked.connect(self._clear_log)
        hdr.addWidget(self.clear_btn)
        layout.addLayout(hdr)

        # ── Legend ──────────────────────────────────────────────────────────
        legend_row = QHBoxLayout()
        legend_row.setSpacing(8)
        for src, colour in _SOURCE_COLOURS.items():
            dot = QLabel(f"{_SOURCE_ICONS[src]} {src.upper()}")
            dot.setStyleSheet(f"color: {colour}; font-size: 8px; font-weight: 700;")
            legend_row.addWidget(dot)
        legend_row.addStretch()
        layout.addLayout(legend_row)

        # ── Scroll area ──────────────────────────────────────────────────────
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self.log_widget = QWidget()
        self.log_layout = QVBoxLayout(self.log_widget)
        self.log_layout.setContentsMargins(0, 0, 0, 0)
        self.log_layout.setSpacing(4)
        self.log_layout.addStretch()

        self.scroll.setWidget(self.log_widget)
        layout.addWidget(self.scroll)

        # Empty state
        self._empty_label = QLabel("No context events yet — start a session.")
        self._empty_label.setStyleSheet("color: #3b4266; font-style: italic; font-size: 10px;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.log_layout.insertWidget(0, self._empty_label)

    # ── Public API ────────────────────────────────────────────────────────────

    def activate(self):
        """Called when the timeline tab is brought into view."""
        self._refresh()

    def deactivate(self):
        """Called when timeline goes off screen — pause polling to save CPU."""
        self._refresh_timer.stop()

    def resume(self):
        self._refresh_timer.start(2000)

    # ── Private ───────────────────────────────────────────────────────────────

    def _refresh(self):
        """Pull new events from Nexus and append them to the log."""
        try:
            entries = self._get_nexus_events()
        except Exception as e:
            logger.debug(f"Timeline refresh error: {e}")
            return

        if len(entries) == self._last_entry_count:
            return  # nothing new

        # Only append new entries (don't full-redraw)
        new_entries = entries[self._last_entry_count:]
        self._last_entry_count = len(entries)

        if new_entries and self._empty_label.isVisible():
            self._empty_label.hide()

        for ev in new_entries:
            self.log_layout.insertWidget(
                self.log_layout.count() - 1,
                self._make_event_card(ev),
            )

        # Auto-scroll to bottom
        QTimer.singleShot(30, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        ))

    def _get_nexus_events(self):
        """Pull the flat event list from Nexus (returns list of dicts)."""
        if hasattr(self.nexus, "get_timeline"):
            return self.nexus.get_timeline() or []
        # Fallback: reconstruct a minimal timeline from the snapshot
        snap = self.nexus.get_snapshot() if hasattr(self.nexus, "get_snapshot") else {}
        events = []
        if snap.get("latest_ocr"):
            events.append({"source": "screen", "text": snap["latest_ocr"][:120], "ts": 0})
        if snap.get("full_audio_history"):
            events.append({"source": "audio", "text": snap["full_audio_history"][-120:], "ts": 0})
        return events

    def _make_event_card(self, ev: dict) -> QFrame:
        source = ev.get("source", "manual")
        colour = _SOURCE_COLOURS.get(source, "#94a3b8")
        icon = _SOURCE_ICONS.get(source, "•")
        text = (ev.get("text") or "")[:140]
        ts = ev.get("ts") or 0

        card = QFrame()
        card.setStyleSheet(
            f"background: rgba(20,20,40,180); border-left: 2px solid {colour};"
            " border-radius: 4px; margin: 0px;"
        )
        hl = QHBoxLayout(card)
        hl.setContentsMargins(8, 4, 8, 4)
        hl.setSpacing(6)

        src_lbl = QLabel(f"{icon}")
        src_lbl.setFixedWidth(16)
        src_lbl.setStyleSheet(f"color: {colour}; font-size: 12px;")
        hl.addWidget(src_lbl)

        body = QLabel(text)
        body.setWordWrap(True)
        body.setStyleSheet("color: #9ba3c0; font-size: 10px;")
        hl.addWidget(body, 1)

        if ts:
            ts_str = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            ts_lbl = QLabel(ts_str)
            ts_lbl.setStyleSheet("color: #3b4266; font-size: 9px;")
            ts_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            hl.addWidget(ts_lbl)

        return card

    def _clear_log(self):
        """Remove all rendered event cards from the layout."""
        while self.log_layout.count() > 1:
            item = self.log_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._last_entry_count = 0
        self._empty_label.show()
