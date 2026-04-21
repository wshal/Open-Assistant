from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath
from PyQt6.QtCore import Qt, QRect

class PremiumCheckBox(QCheckBox):
    """
    A custom-painted checkbox that guarantees tick visibility using direct QPainter vector drawing.
    Bypasses the unreliable QSS image rendering for indicators.
    """
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # We handle geometry manually, so we keep the base style clean
        self.setStyleSheet("background: transparent; border: none;")
        self.setMinimumHeight(22)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Layout metrics
        indicator_size = 18
        spacing = 10
        rect = self.rect()
        
        # Calculate centering for the box
        box_y = (rect.height() - indicator_size) // 2
        indicator_rect = QRect(0, box_y, indicator_size, indicator_size)
        text_rect = QRect(indicator_size + spacing, 0, rect.width() - indicator_size - spacing, rect.height())
        
        # 1. Draw the box (Indicator)
        if self.isChecked():
            bg_color = QColor("#4f46e5")
            border_color = QColor("#6366f1")
        else:
            bg_color = QColor(30, 30, 60, 200) # Semi-dark transparent
            border_color = QColor(120, 130, 255, 60)
            
        # Highlight on hover
        if self.underMouse():
            border_color = QColor("#818cf8")
            if not self.isChecked():
                bg_color = QColor(45, 45, 90, 220)
            
        painter.setPen(QPen(border_color, 1.2))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(indicator_rect, 4, 4)
        
        # 2. Draw the Checkmark (Manual Vector Path)
        if self.isChecked():
            tick_pen = QPen(Qt.GlobalColor.white, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(tick_pen)
            
            # Draw a sharp white checkmark
            path = QPainterPath()
            ix = indicator_rect.x()
            iy = indicator_rect.y()
            
            # Start: Middle-Left | Peak: Bottom-Middle | End: Top-Right
            path.moveTo(ix + 4.5, iy + 9.5)
            path.lineTo(ix + 8.5, iy + 13.5)
            path.lineTo(ix + 14.5, iy + 5.5)
            painter.drawPath(path)
            
        # 3. Draw the Label Text
        painter.setPen(QColor("#cbd5e1" if not self.isChecked() else "#f1f5f9"))
        font = self.font()
        font.setPixelSize(11)
        painter.setFont(font)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())
        
        painter.end()
