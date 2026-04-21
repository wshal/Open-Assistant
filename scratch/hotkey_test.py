from pynput import keyboard
import time

print("--- Hotkey Diagnostic Tool ---")
print("Press any key to see its representation in pynput.")
print("Press Ctrl+C to exit.\n")

def on_press(key):
    try:
        print(f"Key: {key} | Type: {type(key)} | Char: {getattr(key, 'char', 'N/A')} | VK: {getattr(key, 'vk', 'N/A')}")
    except Exception as e:
        print(f"Error: {e}")

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
