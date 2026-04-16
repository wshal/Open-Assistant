# -*- mode: python ; coding: utf-8 -*-
import os

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config.yaml', '.'),
        ('assets', 'assets'),
    ],
    hiddenimports=[
        'google.genai', 'groq', 'mistralai', 'cohere', 'together',
        'anthropic', 'ollama', 'faster_whisper', 'chromadb',
        'rapidocr_onnxruntime', 'sounddevice', 'pynput',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='OpenAssist-AI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='assets/icon.ico',
)

coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=True,
    name='OpenAssist-AI',
)