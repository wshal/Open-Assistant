"""Ghost Cursor — stealth cursor replacement for screen-share protection.

Instead of creating a separate top-level window (which ALWAYS has a visible
DWM compositor surface on Windows, causing the "transparent box" artifact),
this module creates a **child widget** that lives INSIDE the overlay.

As a child widget:
  - It's part of the overlay's existing HWND — no new DWM surface, no box.
  - It inherits the overlay's WDA_EXCLUDEFROMCAPTURE automatically.
  - WA_TransparentForMouseEvents lets clicks pass through to buttons below.

Result:
  - Local user sees a cursor arrow drawn on the overlay surface.
  - Remote screen-share viewers see nothing (overlay is capture-excluded).
"""

import sys
import ctypes
from ctypes import Structure, c_bool, c_ulong, c_void_p, c_uint32, c_int32, c_uint16, POINTER

from PyQt6.QtWidgets import QWidget, QFrame, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, QPoint, QPointF, QEvent, QObject
from PyQt6.QtGui import QPainter, QColor, QPen, QPolygonF, QBrush, QCursor, QImage, QPixmap, QBitmap, QRegion

from utils.logger import setup_logger

logger = setup_logger(__name__)


# ── Windows API Structures ──────────────────────────────────────────────────

class ICONINFO(Structure):
    _fields_ = [
        ("fIcon", c_bool),
        ("xHotspot", c_ulong),
        ("yHotspot", c_ulong),
        ("hbmMask", c_void_p),
        ("hbmColor", c_void_p)
    ]


class BITMAPINFOHEADER(Structure):
    _fields_ = [
        ("biSize", c_uint32),
        ("biWidth", c_int32),
        ("biHeight", c_int32),
        ("biPlanes", c_uint16),
        ("biBitCount", c_uint16),
        ("biCompression", c_uint32),
        ("biSizeImage", c_uint32),
        ("biXPelsPerMeter", c_int32),
        ("biYPelsPerMeter", c_int32),
        ("biClrUsed", c_uint32),
        ("biClrImportant", c_uint32),
    ]


class BITMAPINFO(Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", c_uint32 * 3)
    ]


def get_windows_default_cursor(device_pixel_ratio: float = 1.0) -> tuple[QPixmap, QPoint]:
    """Loads the native Windows default arrow cursor, strips its shadow, crops it to the minimal bounding box,
    and returns it as a QPixmap and its adjusted hotspot.
    """
    if sys.platform != "win32":
        return QPixmap(), QPoint(0, 0)

    try:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        # Configure ctypes arguments and return types to prevent overflow on 64-bit
        user32.LoadCursorW.argtypes = [c_void_p, c_void_p]
        user32.LoadCursorW.restype = c_void_p

        user32.GetSystemMetrics.argtypes = [c_int32]
        user32.GetSystemMetrics.restype = c_int32

        user32.GetIconInfo.argtypes = [c_void_p, POINTER(ICONINFO)]
        user32.GetIconInfo.restype = c_bool

        gdi32.DeleteObject.argtypes = [c_void_p]
        gdi32.DeleteObject.restype = c_bool

        user32.GetDC.argtypes = [c_void_p]
        user32.GetDC.restype = c_void_p

        user32.ReleaseDC.argtypes = [c_void_p, c_void_p]
        user32.ReleaseDC.restype = c_int32

        gdi32.CreateCompatibleDC.argtypes = [c_void_p]
        gdi32.CreateCompatibleDC.restype = c_void_p

        gdi32.DeleteDC.argtypes = [c_void_p]
        gdi32.DeleteDC.restype = c_bool

        gdi32.SelectObject.argtypes = [c_void_p, c_void_p]
        gdi32.SelectObject.restype = c_void_p

        gdi32.CreateDIBSection.argtypes = [c_void_p, POINTER(BITMAPINFO), c_uint32, POINTER(c_void_p), c_void_p, c_uint32]
        gdi32.CreateDIBSection.restype = c_void_p

        user32.DrawIconEx.argtypes = [c_void_p, c_int32, c_int32, c_void_p, c_int32, c_int32, c_uint32, c_void_p, c_uint32]
        user32.DrawIconEx.restype = c_bool

        # Load default arrow cursor
        IDC_ARROW = 32512
        hcursor = user32.LoadCursorW(None, IDC_ARROW)
        if not hcursor:
            return QPixmap(), QPoint(0, 0)

        # Get system cursor metrics (SM_CXCURSOR=13, SM_CYCURSOR=14)
        SM_CXCURSOR = 13
        SM_CYCURSOR = 14
        width = user32.GetSystemMetrics(SM_CXCURSOR) or 32
        height = user32.GetSystemMetrics(SM_CYCURSOR) or 32

        # Get hotspot
        icon_info = ICONINFO()
        hotspot = QPoint(0, 0)
        if user32.GetIconInfo(hcursor, ctypes.byref(icon_info)):
            hotspot = QPoint(int(icon_info.xHotspot), int(icon_info.yHotspot))
            if icon_info.hbmMask:
                gdi32.DeleteObject(icon_info.hbmMask)
            if icon_info.hbmColor:
                gdi32.DeleteObject(icon_info.hbmColor)

        # Draw the cursor into a 32-bit DIB section with alpha support
        hdc_screen = user32.GetDC(None)
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height  # top-down bitmap
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0  # BI_RGB

        pixel_data = c_void_p()
        hbitmap = gdi32.CreateDIBSection(
            hdc_mem,
            ctypes.byref(bmi),
            0,
            ctypes.byref(pixel_data),
            None,
            0
        )

        if not hbitmap or not pixel_data:
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(None, hdc_screen)
            return QPixmap(), QPoint(0, 0)

        old_bitmap = gdi32.SelectObject(hdc_mem, hbitmap)

        # Clear memory buffer to fully transparent
        ctypes.memset(pixel_data, 0, width * height * 4)

        # Draw the system cursor onto the DIB section
        DI_NORMAL = 0x0003
        user32.DrawIconEx(hdc_mem, 0, 0, hcursor, width, height, 0, None, DI_NORMAL)

        # Read the raw byte data and convert to QImage, then make a deep copy immediately
        buf = ctypes.string_at(pixel_data, width * height * 4)
        qimg = QImage(buf, width, height, QImage.Format.Format_ARGB32).copy()

        # Cleanup GDI handles immediately
        gdi32.SelectObject(hdc_mem, old_bitmap)
        gdi32.DeleteObject(hbitmap)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(None, hdc_screen)

        # Threshold alpha channel to remove soft shadows and antialiasing anomalies
        qimg = qimg.convertToFormat(QImage.Format.Format_ARGB32)
        for y in range(height):
            for x in range(width):
                color = qimg.pixelColor(x, y)
                if color.alpha() < 100:
                    color.setAlpha(0)
                else:
                    color.setAlpha(255)
                qimg.setPixelColor(x, y, color)

        # Find the bounding box of non-transparent pixels to crop empty space
        min_x = width
        max_x = 0
        min_y = height
        max_y = 0
        for y in range(height):
            for x in range(width):
                if qimg.pixelColor(x, y).alpha() > 0:
                    if x < min_x: min_x = x
                    if x > max_x: max_x = x
                    if y < min_y: min_y = y
                    if y > max_y: max_y = y

        if max_x >= min_x and max_y >= min_y:
            cropped_w = max_x - min_x + 1
            cropped_h = max_y - min_y + 1
            qimg = qimg.copy(min_x, min_y, cropped_w, cropped_h)
            # Adjust hotspot relative to cropped image
            hotspot = QPoint(hotspot.x() - min_x, hotspot.y() - min_y)
        else:
            return QPixmap(), QPoint(0, 0)

        pixmap = QPixmap.fromImage(qimg)
        if device_pixel_ratio != 1.0:
            pixmap.setDevicePixelRatio(device_pixel_ratio)
            hotspot = QPoint(int(hotspot.x() / device_pixel_ratio), int(hotspot.y() / device_pixel_ratio))

        return pixmap, hotspot

    except Exception as e:
        logger.warning("Error loading native Windows default cursor: %s", e)
        return QPixmap(), QPoint(0, 0)


def apply_native_window_styles(hwnd: int):
    """Enforces WS_POPUP style and layering flags directly on the HWND to bypass default frames and shadow artifacts."""
    if sys.platform != "win32":
        return
    try:
        user32 = ctypes.windll.user32
        
        # Configure SetWindowLongPtrW or SetWindowLongW depending on 64-bit/32-bit availability
        if hasattr(user32, "SetWindowLongPtrW"):
            user32.SetWindowLongPtrW.argtypes = [c_void_p, c_int32, c_void_p]
            user32.SetWindowLongPtrW.restype = c_void_p
            set_long = user32.SetWindowLongPtrW
        else:
            user32.SetWindowLongW.argtypes = [c_void_p, c_int32, c_int32]
            user32.SetWindowLongW.restype = c_int32
            set_long = user32.SetWindowLongW

        user32.SetWindowPos.argtypes = [c_void_p, c_void_p, c_int32, c_int32, c_int32, c_int32, c_uint32]
        user32.SetWindowPos.restype = c_bool

        # Basic style: WS_POPUP (0x80000000) (no titlebar, borders, system menus)
        WS_POPUP = 0x80000000
        WS_CLIPSIBLINGS = 0x04000000
        WS_CLIPCHILDREN = 0x02000000
        style = WS_POPUP | WS_CLIPSIBLINGS | WS_CLIPCHILDREN
        set_long(c_void_p(hwnd), -16, c_void_p(style) if hasattr(user32, "SetWindowLongPtrW") else style)

        # Extended style: WS_EX_LAYERED (0x00080000), WS_EX_TRANSPARENT (0x00000020), WS_EX_NOACTIVATE (0x08000000), WS_EX_TOOLWINDOW (0x00000080), WS_EX_TOPMOST (0x00000008)
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_NOACTIVATE = 0x08000000
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_TOPMOST = 0x00000008
        ex_style = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST
        set_long(c_void_p(hwnd), -20, c_void_p(ex_style) if hasattr(user32, "SetWindowLongPtrW") else ex_style)

        # Apply frame changes immediately (SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE)
        user32.SetWindowPos(c_void_p(hwnd), None, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020)
    except Exception as e:
        logger.debug("Failed to apply native styles on HWND %s: %s", hwnd, e)


def disable_dwm_artifacts(hwnd: int):
    """Disables DWM shadow, transitions, rounded corners, and non-client rendering to prevent transparent/black boxes in screen share."""
    if sys.platform != "win32":
        return
    try:
        dwmapi = ctypes.WinDLL("dwmapi")
        dwmapi.DwmSetWindowAttribute.argtypes = [c_void_p, c_uint32, c_void_p, c_uint32]
        dwmapi.DwmSetWindowAttribute.restype = c_int32

        # Disable shadow/non-client rendering (DWMWA_NCRENDERING_POLICY = 2, DWMNCRP_DISABLED = 1)
        policy = c_int32(1)
        dwmapi.DwmSetWindowAttribute(
            c_void_p(hwnd),
            2,
            ctypes.byref(policy),
            ctypes.sizeof(policy)
        )

        # Disable transitions (DWMWA_TRANSITIONS_FORCEDISABLED = 3)
        transitions = c_int32(1)
        dwmapi.DwmSetWindowAttribute(
            c_void_p(hwnd),
            3,
            ctypes.byref(transitions),
            ctypes.sizeof(transitions)
        )

        # Disable rounded corners on Windows 11 (DWMWA_WINDOW_CORNER_PREFERENCE = 33, DWMWCP_DONOTROUND = 1)
        corner_pref = c_int32(1)
        dwmapi.DwmSetWindowAttribute(
            c_void_p(hwnd),
            33,
            ctypes.byref(corner_pref),
            ctypes.sizeof(corner_pref)
        )

        # Exclude from peek (DWMWA_EXCLUDED_FROM_PEEK = 12)
        peek = c_int32(1)
        dwmapi.DwmSetWindowAttribute(
            c_void_p(hwnd),
            12,
            ctypes.byref(peek),
            ctypes.sizeof(peek)
        )
    except Exception as e:
        logger.debug("Failed to disable DWM artifacts on HWND %s: %s", hwnd, e)


# ── Ghost Cursor Child Widget ───────────────────────────────────────────────

class GhostCursorWidget(QWidget):
    """Transparent child widget that draws a cursor sprite inside its parent.

    Because this is a child widget (not a top-level window), it has no DWM
    surface and no "transparent box". It inherits the parent overlay's
    capture-exclusion automatically.
    """

    def __init__(self, parent):
        super().__init__(parent)
        # Let mouse events pass through to buttons/input underneath
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setGeometry(parent.rect())
        self.hide()

        # Classic Windows arrow cursor polygon (fallback)
        self._fallback_polygon = QPolygonF([
            QPointF(0, 0),
            QPointF(0, 20),
            QPointF(4.5, 16),
            QPointF(7.5, 23),
            QPointF(10, 22),
            QPointF(7, 15),
            QPointF(12, 15),
            QPointF(0, 0),
        ])

        self._pixmap = QPixmap()
        self._hotspot = QPoint(0, 0)
        self._active = False
        self._frozen = False
        self._last_cursor_pos = QPoint(-1, -1)
        self._draw_pos = QPoint(0, 0)
        self._freeze_pos = QPoint(0, 0)

        # Repaint timer — triggers update() at ~60fps to redraw cursor position
        self._repaint_timer = QTimer(self)
        self._repaint_timer.setInterval(16)
        self._repaint_timer.timeout.connect(self._on_tick)
        self._freeze_timer = QTimer(self)
        self._freeze_timer.setSingleShot(True)
        self._freeze_timer.timeout.connect(self._end_freeze)

    def activate(self):
        """Show the ghost cursor and start tracking."""
        if self.parent() is None:
            return
        if self._freeze_timer.isActive():
            self._freeze_timer.stop()
        if not self._active:
            self._active = True
            self.show()
        self._frozen = False
        self._last_cursor_pos = QPoint(-1, -1)

        # Load/update cursor pixmap with current device pixel ratio
        dpr = self.devicePixelRatioF()
        self._pixmap, self._hotspot = get_windows_default_cursor(dpr)

        # Match parent size and bring to front of all siblings
        self._sync_geometry()
        self.raise_()
        self._draw_pos = self._resolve_draw_pos()
        self._repaint_timer.start()
        self.update()
        logger.debug("Ghost cursor activated (child widget)")

    def deactivate(self):
        """Hide the ghost cursor and stop tracking."""
        if not self._active:
            return
        self._freeze_timer.stop()
        self._active = False
        self._frozen = False
        self._repaint_timer.stop()
        self.hide()
        logger.debug("Ghost cursor deactivated")

    def freeze(self, duration_ms: int = 180):
        """Freeze the cursor at the last in-bounds point for a short boundary hold."""
        if not self._active:
            return
        self._freeze_pos = QPoint(self._draw_pos)
        self._frozen = True
        self._repaint_timer.stop()
        self.show()
        self.raise_()
        self.update()
        if duration_ms > 0:
            self._freeze_timer.start(duration_ms)
        else:
            self._end_freeze()

    def _end_freeze(self):
        if not self._active:
            return
        self._freeze_timer.stop()
        self._frozen = False
        self._active = False
        self.hide()
        logger.debug("Ghost cursor boundary freeze ended")

    def _sync_geometry(self):
        """Sync our size to the parent's rect."""
        p = self.parent()
        if p is None:
            return
        target = p.rect()
        if self.size() != target.size():
            self.setGeometry(target)

    def _resolve_draw_pos(self) -> QPoint:
        """Return the current cursor position, clamped to the parent bounds."""
        current_pos = self.mapFromGlobal(QCursor.pos())
        parent = self.parent()
        if parent is None:
            return current_pos
        rect = parent.rect()
        x = max(0, min(current_pos.x(), rect.width() - 1))
        y = max(0, min(current_pos.y(), rect.height() - 1))
        if rect.contains(current_pos):
            return QPoint(current_pos)
        return QPoint(x, y)

    def _on_tick(self):
        """Called every 16ms — sync geometry, maintain z-order, repaint if
        the cursor has moved."""
        try:
            self._sync_geometry()
            # Keep above any widgets the parent may have created after us
            self.raise_()
            if self._frozen:
                return
            current_pos = QCursor.pos()
            if current_pos != self._last_cursor_pos:
                self._last_cursor_pos = current_pos
                self._draw_pos = self._resolve_draw_pos()
                self.update()
        except RuntimeError:
            # Widget or parent is being destroyed — stop the timer gracefully
            self._repaint_timer.stop()

    def paintEvent(self, event):
        """Draw the cursor arrow at the current mouse position."""
        if not self._active and not self._frozen:
            return

        local_pos = self._freeze_pos if self._frozen else self._draw_pos

        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if not self._pixmap.isNull():
            # Draw native Windows default cursor image
            painter.drawPixmap(local_pos - self._hotspot, self._pixmap)
        else:
            # Draw fallback polygon
            painter.translate(local_pos)
            painter.setPen(QPen(QColor(0, 0, 0, 230), 1.4))
            painter.setBrush(QBrush(QColor(255, 255, 255, 250)))
            painter.drawPolygon(self._fallback_polygon)

        painter.end()

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_frozen(self) -> bool:
        return self._frozen


# ── Static Cursor Overlay (Top-level Window) ────────────────────────────────

class StaticCursorOverlay(QWidget):
    """A tiny, top-level window that is NOT excluded from capture.
    It draws a static cursor at the point of entry so screen-share viewers
    see the cursor 'parked' at the edge of the window instead of disappearing.
    """

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.ToolTip
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        # Classic Windows arrow cursor polygon (fallback)
        self._fallback_polygon = QPolygonF([
            QPointF(0, 0),
            QPointF(0, 20),
            QPointF(4.5, 16),
            QPointF(7.5, 23),
            QPointF(10, 22),
            QPointF(7, 15),
            QPointF(12, 15),
            QPointF(0, 0),
        ])

        self._pixmap = QPixmap()
        self._hotspot = QPoint(0, 0)
        self.update_cursor()

        # Enforce native WS_POPUP styles and DWM shadow/corner disabling immediately
        hwnd = int(self.winId())
        if hwnd != 0:
            apply_native_window_styles(hwnd)
            disable_dwm_artifacts(hwnd)

    def showEvent(self, event):
        """Re-apply styles and DWM properties when shown to prevent Qt handle recreation from wiping them."""
        super().showEvent(event)
        hwnd = int(self.winId())
        if hwnd != 0:
            apply_native_window_styles(hwnd)
            disable_dwm_artifacts(hwnd)

    def update_cursor(self):
        """Update/reload the native cursor image matching the current DPI."""
        dpr = self.devicePixelRatioF()
        self._pixmap, self._hotspot = get_windows_default_cursor(dpr)
        if not self._pixmap.isNull():
            self.setFixedSize(self._pixmap.width(), self._pixmap.height())
            # Set window mask to exclude all transparent pixels outside the cursor image shape
            self.setMask(self._pixmap.mask())
        else:
            self.setFixedSize(32, 32)
            # Use fallback polygon to set mask
            polygon_i = self._fallback_polygon.toPolygon()
            self.setMask(QRegion(polygon_i))

    def hotspot(self) -> QPoint:
        """Returns the hotspot coordinates offset."""
        return self._hotspot

    def paintEvent(self, event):
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        
        if not self._pixmap.isNull():
            painter.drawPixmap(0, 0, self._pixmap)
        else:
            painter.setPen(QPen(QColor(0, 0, 0, 230), 1.4))
            painter.setBrush(QBrush(QColor(255, 255, 255, 250)))
            painter.drawPolygon(self._fallback_polygon)
            
        painter.end()


# ── Stealth Tooltip Child Widget ────────────────────────────────────────────

class StealthToolTip(QFrame):
    """Custom tooltip widget drawn as a child of the capture-excluded overlay.
    Because it is a child widget, it is 100% hidden from screen capture.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(20, 20, 20, 240);
                border: 1px solid rgba(80, 80, 80, 160);
                border-radius: 4px;
            }
            QLabel {
                color: #e2e2e2;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
                background: transparent;
                padding: 4px 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.hide()

    def show_tip(self, text: str, pos: QPoint):
        self.label.setText(text)
        self.adjustSize()
        p = self.parent()
        if p is not None:
            # Position tooltip offset from the cursor position
            x = pos.x() + 15
            y = pos.y() + 15
            # Constrain to parent boundaries
            if x + self.width() > p.width():
                x = p.width() - self.width() - 5
            if y + self.height() > p.height():
                y = pos.y() - self.height() - 5
            self.move(QPoint(max(0, int(x)), max(0, int(y))))
        else:
            self.move(pos)
        self.raise_()
        self.show()


# ── Global Stealth Tooltip Event Filter ──────────────────────────────────────

class GlobalStealthToolTipFilter(QObject):
    """Global application event filter that:
    1. Intercepts all QEvent.Type.ToolTip events on overlay windows, redirecting
       them to our custom StealthToolTip to prevent native OS tooltip leaks.
    2. Intercepts QEvent.Type.Show events on any top-level window to automatically
       apply SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) when stealth is enabled,
       preventing capture leaks from combo box popups, dropdowns, and menus.
    """

    def __init__(self, parent_app):
        super().__init__(parent_app)
        self.app = parent_app if hasattr(parent_app, "config") else None
        self.tooltips = {}      # window_id -> StealthToolTip
        self.hide_timers = {}   # window_id -> QTimer

    def eventFilter(self, obj, event):
        try:
            # 1. Capture-exclude any top-level popup or window when it is shown
            if event.type() == QEvent.Type.Show:
                if isinstance(obj, QWidget) and obj.isWindow():
                    win_class_name = type(obj).__name__
                    # Never exclude the StaticCursorOverlay which is intended to be captured
                    if win_class_name != "StaticCursorOverlay" and sys.platform == "win32":
                        try:
                            hwnd = int(obj.winId())
                            if hwnd != 0:
                                user32 = ctypes.windll.user32
                                root_hwnd = user32.GetAncestor(hwnd, 2) or hwnd  # GA_ROOT = 2
                                stealth_enabled = True
                                if self.app is not None:
                                    stealth_enabled = self.app.config.get("stealth.enabled", True)
                                
                                affinity = 0x00000011 if stealth_enabled else 0x00000000
                                user32.SetWindowDisplayAffinity(root_hwnd, affinity)
                        except Exception as e:
                            logger.debug("Failed to apply auto display affinity on show: %s", e)

            # 2. Intercept ToolTip events
            elif event.type() == QEvent.Type.ToolTip:
                if not isinstance(obj, QWidget):
                    return False
                window = obj.window()
                if window is None:
                    return False

                win_class_name = type(window).__name__
                is_overlay = win_class_name in (
                    "OverlayWindow",
                    "MiniOverlayWindow",
                    "Overlay",
                    "MiniOverlay",
                )

                if is_overlay:
                    # Prevent native tooltip window
                    event.accept()

                    text = obj.toolTip()
                    if text:
                        wid = id(window)
                        tooltip = self.tooltips.get(wid)
                        # Check if tooltip widget has been destroyed / is deleted C++ object
                        try:
                            if tooltip is not None and tooltip.parent() is None:
                                tooltip = None
                        except RuntimeError:
                            tooltip = None

                        if tooltip is None:
                            tooltip = StealthToolTip(window)
                            self.tooltips[wid] = tooltip

                            timer = QTimer(window)
                            timer.setSingleShot(True)
                            timer.timeout.connect(tooltip.hide)
                            self.hide_timers[wid] = timer

                        timer = self.hide_timers[wid]

                        # Map global cursor coordinates to parent window local coordinates
                        global_pos = QCursor.pos()
                        local_pos = window.mapFromGlobal(global_pos)
                        tooltip.show_tip(text, local_pos)
                        timer.start(4000)  # auto-hide after 4 seconds
                    return True  # filter event, do not propagate to OS tooltip

            elif event.type() in (QEvent.Type.Leave, QEvent.Type.MouseButtonPress):
                # Hide tooltip when mouse leaves or clicks
                if isinstance(obj, QWidget):
                    window = obj.window()
                    if window is not None:
                        wid = id(window)
                        tooltip = self.tooltips.get(wid)
                        if tooltip is not None:
                            try:
                                tooltip.hide()
                                self.hide_timers[wid].stop()
                            except RuntimeError:
                                pass
        except Exception as e:
            logger.exception("Error in GlobalStealthToolTipFilter.eventFilter: %s", e)

        return super().eventFilter(obj, event)
