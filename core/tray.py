"""System tray icon with quick actions."""

from PyQt6.QtWidgets import QSystemTrayIcon, QMenu, QApplication
from PyQt6.QtGui import QIcon, QAction, QPixmap, QPainter, QColor
from PyQt6.QtCore import Qt


class SystemTray:
    def __init__(self, app):
        self.app = app
        self.tray = QSystemTrayIcon()
        self.tray.setIcon(self._make_icon())
        self.tray.setToolTip("OpenAssist AI")

        menu = QMenu()
        menu.setStyleSheet(
            """
            QMenu { background: #1a1a2e; color: #c0c0dd; border: 1px solid #333355; }
            QMenu::item:selected { background: #2a2a4e; }
            """
        )

        show_action = QAction("Show/Hide Overlay", menu)
        show_action.triggered.connect(app.toggle_overlay)
        menu.addAction(show_action)
        menu.addSeparator()

        modes = [
            ("General", "general"),
            ("Interview", "interview"),
            ("Meeting", "meeting"),
            ("Coding", "coding"),
            ("Writing", "writing"),
            ("Exam", "exam"),
        ]
        mode_menu = menu.addMenu("Switch Mode")
        for label, name in modes:
            action = QAction(label, mode_menu)
            action.triggered.connect(lambda _, n=name: app.switch_mode(n))
            mode_menu.addAction(action)

        menu.addSeparator()

        quick = QAction("Quick Answer (Screen)", menu)
        quick.triggered.connect(app.quick_answer)
        menu.addAction(quick)

        stealth = QAction("Reapply Stealth", menu)
        stealth.triggered.connect(app.toggle_stealth_mode)
        menu.addAction(stealth)

        settings = QAction("Settings", menu)
        settings.triggered.connect(self._open_settings)
        menu.addAction(settings)

        menu.addSeparator()

        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(QApplication.quit)
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_click)
        self.tray.show()

    def _on_click(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.app.toggle_overlay()

    def _open_settings(self):
        self.app.open_settings()

    @staticmethod
    def _make_icon() -> QIcon:
        """Generate a simple app icon."""
        pixmap = QPixmap(64, 64)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(70, 70, 170))
        painter.setPen(QColor(100, 100, 200))
        painter.drawRoundedRect(4, 4, 56, 56, 12, 12)
        painter.setPen(QColor(220, 220, 255))
        font = painter.font()
        font.setPixelSize(28)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "OA")
        painter.end()
        return QIcon(pixmap)
