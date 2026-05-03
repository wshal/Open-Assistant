import unittest
import sys
from unittest.mock import patch, MagicMock

# Ensure we can import from core/stealth/capture
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture.audio import AudioCapture
from stealth.anti_detect import StealthManager

class TestMacOSNativeParity(unittest.TestCase):

    @patch('sys.platform', 'darwin')
    @patch('stealth.anti_detect.logger')
    def test_macos_stealth_bridge_invocation(self, mock_logger):
        """Test that macOS stealth logic attempts to use AppKit via pyobjc."""
        # Create a mock window with winId returning a pointer (represented as an int)
        mock_window = MagicMock()
        mock_window.winId.return_value = 12345678
        
        manager = StealthManager({})
        
        # We patch 'objc' in sys.modules to simulate pyobjc being installed
        # and test if it interacts with the AppKit NSWindow correctly.
        mock_objc = MagicMock()
        mock_appkit = MagicMock()
        
        # Setup the mock NSWindow and NSView
        mock_ns_window = MagicMock()
        mock_ns_view = MagicMock()
        mock_ns_view.window.return_value = mock_ns_window
        mock_objc.objc_object.return_value = mock_ns_view
        
        with patch.dict('sys.modules', {'objc': mock_objc, 'AppKit': mock_appkit}):
            # Set the required constants on the mock AppKit module
            mock_appkit.NSWindowCollectionBehaviorCanJoinAllSpaces = 1
            mock_appkit.NSWindowCollectionBehaviorMoveToActiveSpace = 2
            mock_appkit.NSWindowCollectionBehaviorTransient = 8
            
            manager.apply_to_window(mock_window, enabled=True)
            
            # Assert core fallback Qt flag was set
            mock_window.setWindowFlag.assert_called_with(
                mock_window.setWindowFlag.call_args[0][0], True
            )
            
            # Assert PyObjC bridge was called
            mock_objc.objc_object.assert_called_with(c_void_p=12345678)
            mock_ns_view.window.assert_called_once()
            
            # Assert collection behavior was set
            mock_ns_window.setCollectionBehavior_.assert_called_with(1 | 2 | 8)

    @patch('sys.platform', 'darwin')
    @patch('capture.audio.AudioCapture._start_macos_system_audio')
    @patch.dict('sys.modules', {'sounddevice': MagicMock()})
    def test_macos_audio_branching(self, mock_start_macos):
        """Test that audio capture loop cleanly branches to macOS binary loopback."""
        config = {"capture.audio.mode": "system", "capture.audio.enabled": True}
        
        capture = AudioCapture(config)
        capture._running = True
        
        # Stop the loop early to just test the initialization branch
        def _mock_start():
            capture._running = False
            return True
            
        mock_start_macos.side_effect = _mock_start
        
        capture._capture_loop()
        
        # Verify it skipped WASAPI and branched straight to macOS system audio
        mock_start_macos.assert_called_once()

if __name__ == '__main__':
    unittest.main()
