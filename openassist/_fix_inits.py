"""One-time fix script: rewrite all corrupt __init__.py files."""
from pathlib import Path

# ── Empty inits (just need to be valid Python files) ─────────────────────────
empty_inits = [
    'ai/__init__.py',
    'capture/__init__.py',
    'core/__init__.py',
    'stealth/__init__.py',
    'ui/__init__.py',
    'utils/__init__.py',
    'ai/detectors/__init__.py',
]

for path in empty_inits:
    p = Path(path)
    p.write_text('', encoding='utf-8')
    print(f'Fixed (empty): {path}')

# ── providers/__init__.py ─────────────────────────────────────────────────────
Path('ai/providers/__init__.py').write_text(
    '''"""Provider registry - discovers and initialises all AI providers from config."""

from utils.logger import setup_logger

logger = setup_logger(__name__)


def init_providers(config) -> dict:
    """Instantiate every configured provider and return the enabled ones."""
    from ai.providers.groq_provider import GroqProvider
    from ai.providers.gemini_provider import GeminiProvider
    from ai.providers.cerebras_provider import CerebrasProvider
    from ai.providers.mistral_provider import MistralProvider
    from ai.providers.cohere_provider import CohereProvider
    from ai.providers.ollama_provider import OllamaProvider
    from ai.providers.openai_compat import OpenAICompatProvider

    candidates = {
        "groq":       lambda: GroqProvider(config),
        "gemini":     lambda: GeminiProvider(config),
        "cerebras":   lambda: CerebrasProvider(config),
        "sambanova":  lambda: OpenAICompatProvider("sambanova", config),
        "together":   lambda: OpenAICompatProvider("together", config),
        "openrouter": lambda: OpenAICompatProvider("openrouter", config),
        "hyperbolic": lambda: OpenAICompatProvider("hyperbolic", config),
        "mistral":    lambda: MistralProvider(config),
        "cohere":     lambda: CohereProvider(config),
        "ollama":     lambda: OllamaProvider(config),
    }

    try:
        from ai.providers.openai_provider import OpenAIProvider
        candidates["openai"] = lambda: OpenAIProvider(config)
    except ImportError:
        pass

    try:
        from ai.providers.anthropic_provider import AnthropicProvider
        candidates["anthropic"] = lambda: AnthropicProvider(config)
    except ImportError:
        pass

    providers = {}
    for name, factory in candidates.items():
        try:
            prov = factory()
            if prov.enabled:
                providers[name] = prov
        except Exception as exc:
            logger.warning(f"  x {name} failed to load: {exc}")

    active = list(providers.keys())
    if active:
        logger.info(f"Providers active: {', '.join(active)}")
    else:
        logger.warning("No providers active - add API keys in Settings")

    return providers
''',
    encoding='utf-8'
)
print('Fixed (content): ai/providers/__init__.py')

# ── modes/__init__.py ─────────────────────────────────────────────────────────
Path('modes/__init__.py').write_text(
    '''"""Mode manager - loads all built-in modes."""

from modes.base import Mode
from modes.general import GeneralMode
from modes.interview import InterviewMode
from modes.coding import CodingMode
from modes.meeting import MeetingMode
from modes.writing import WritingMode
from modes.exam import ExamMode
from utils.logger import setup_logger

logger = setup_logger(__name__)

_ALL_MODES = [GeneralMode, InterviewMode, CodingMode, MeetingMode, WritingMode, ExamMode]


class ModeManager:
    def __init__(self, config):
        self.config = config
        self._modes = {}
        for cls in _ALL_MODES:
            try:
                m = cls(config)
                self._modes[m.name] = m
            except Exception as e:
                logger.warning(f"Could not load mode {cls.__name__}: {e}")

        default = config.get("modes.default", "general")
        self._current_name = default if default in self._modes else next(iter(self._modes), "general")
        logger.info(f"Modes loaded, active: {self._current_name}")

    @property
    def current(self) -> Mode:
        return self._modes.get(self._current_name, next(iter(self._modes.values())))

    def switch(self, name: str):
        if name in self._modes:
            self._current_name = name
        else:
            logger.warning(f"Unknown mode: {name}")

    def cycle(self):
        keys = list(self._modes)
        idx = keys.index(self._current_name) if self._current_name in keys else 0
        self._current_name = keys[(idx + 1) % len(keys)]

    def auto_detect(self, screen_text="", audio_text="", window_category=""):
        text = (screen_text + " " + audio_text + " " + window_category).lower()
        hints = {
            "interview": ["interview", "tell me about", "strengths", "career"],
            "coding":    ["def ", "class ", "import ", "error:", "traceback"],
            "exam":      ["multiple choice", "answer:", "exam", "quiz"],
            "meeting":   ["agenda", "meeting", "action items", "standup"],
            "writing":   ["paragraph", "essay", "article", "draft"],
        }
        for mode_name, keywords in hints.items():
            if any(kw in text for kw in keywords):
                return mode_name
        return None

    @property
    def all_modes(self):
        return list(self._modes.values())
''',
    encoding='utf-8'
)
print('Fixed (content): modes/__init__.py')

# ── Verify no null bytes remain ───────────────────────────────────────────────
import ast
errors = []
for f in Path('.').rglob('*.py'):
    if 'venv' in str(f) or '__pycache__' in str(f):
        continue
    raw = f.read_bytes()
    if b'\x00' in raw:
        errors.append(f'NULL BYTES: {f}')
        continue
    try:
        ast.parse(raw.decode('utf-8', errors='replace'))
    except SyntaxError as e:
        errors.append(f'SYNTAX ERROR {f}: {e}')

if errors:
    print('\nRemaining errors:')
    for e in errors:
        print(' ', e)
else:
    print('\nAll files clean - no null bytes or syntax errors found.')
