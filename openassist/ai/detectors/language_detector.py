"""Programming language detection for coding mode optimization.

Used by:
  - CodingMode: select best model tier (code vs balanced)
  - Router: prefer code-specialized providers (Mistral Codestral)
  - Prompts: include language-specific instructions
  - UI: show detected language badge
"""

import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class LanguageDetection:
    """Result of language detection."""
    language: str           # "python", "javascript", etc.
    confidence: float       # 0.0 to 1.0
    framework: str = ""     # "react", "django", "flask", etc.
    file_type: str = ""     # ".py", ".js", etc.
    is_markup: bool = False # HTML, CSS, XML, Markdown


# Language detection rules: (language, extension, indicators, framework_hints)
LANGUAGE_RULES: Dict[str, Dict] = {
    "python": {
        "extensions": [".py", ".pyw", ".pyi"],
        "strong": [
            r'^\s*def\s+\w+\s*\(', r'^\s*class\s+\w+.*:', r'^\s*import\s+\w+',
            r'^\s*from\s+\w+\s+import', r'if\s+__name__\s*==', r'^\s*async\s+def\s+',
            r'print\s*\(', r'self\.\w+', r'__init__\s*\(self',
            r'^\s*@\w+', r'^\s*elif\s+', r':\s*$',
        ],
        "weak": [
            r'#.*comment', r'"""\s*', r"'''", r'\.append\(',
            r'\.items\(\)', r'\.keys\(\)', r'len\(',
            r'range\(', r'enumerate\(', r'isinstance\(',
        ],
        "frameworks": {
            "django": [r'from django', r'models\.Model', r'HttpResponse', r'urls\.py'],
            "flask": [r'from flask', r'@app\.route', r'Flask\('],
            "fastapi": [r'from fastapi', r'FastAPI\(', r'@app\.(get|post|put)'],
            "pytorch": [r'import torch', r'nn\.Module', r'tensor\('],
            "pandas": [r'import pandas', r'pd\.DataFrame', r'\.iloc\['],
            "numpy": [r'import numpy', r'np\.array', r'np\.zeros'],
        },
    },
    "javascript": {
        "extensions": [".js", ".jsx", ".mjs", ".cjs"],
        "strong": [
            r'\bfunction\s+\w+\s*\(', r'\bconst\s+\w+\s*=', r'\blet\s+\w+\s*=',
            r'\bvar\s+\w+\s*=', r'=>\s*[{(]', r'console\.\w+\(',
            r'\brequire\s*\(', r'\bmodule\.exports', r'document\.\w+',
            r'window\.\w+', r'\bnew\s+Promise',
        ],
        "weak": [
            r'===', r'!==', r'\.\.\.\w+', r'async\s+function',
            r'\.then\(', r'\.catch\(', r'\.map\(', r'\.filter\(',
        ],
        "frameworks": {
            "react": [r'import React', r'useState\(', r'useEffect\(', r'<\w+\s*/>', r'jsx'],
            "vue": [r'createApp', r'ref\(', r'computed\(', r'\.vue'],
            "node": [r'require\(', r'module\.exports', r'process\.env', r'express\('],
            "next": [r'getServerSideProps', r'getStaticProps', r'next/'],
        },
    },
    "typescript": {
        "extensions": [".ts", ".tsx"],
        "strong": [
            r':\s*(string|number|boolean|void|any|never)\b',
            r'\binterface\s+\w+', r'\btype\s+\w+\s*=',
            r'<\w+(\s*,\s*\w+)*>', r'\bas\s+\w+',
            r'\benum\s+\w+', r'\bnamespace\s+\w+',
            r':\s*\w+\[\]', r'\bReadonly<', r'\bPartial<',
        ],
        "weak": [
            r'\.tsx?$', r'tsconfig', r'@types/',
        ],
        "frameworks": {
            "angular": [r'@Component', r'@Injectable', r'@NgModule', r'angular'],
            "react": [r'React\.FC', r'useState<', r'useEffect'],
            "nest": [r'@Controller', r'@Get\(', r'@Post\(', r'NestFactory'],
        },
    },
    "java": {
        "extensions": [".java"],
        "strong": [
            r'public\s+class\s+\w+', r'public\s+static\s+void\s+main',
            r'System\.out\.print', r'@Override', r'\bimplements\s+',
            r'\bextends\s+', r'private\s+\w+\s+\w+', r'protected\s+',
            r'\bnew\s+\w+\(', r'import\s+java\.',
        ],
        "weak": [
            r'\.class\b', r'try\s*\{', r'catch\s*\(\w+', r'throws\s+\w+',
            r'@\w+\(', r'\.stream\(\)',
        ],
        "frameworks": {
            "spring": [r'@SpringBootApplication', r'@RestController', r'@Autowired', r'spring'],
            "android": [r'import android', r'Activity', r'onCreate', r'R\.layout'],
        },
    },
    "cpp": {
        "extensions": [".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"],
        "strong": [
            r'#include\s*<\w+>', r'#include\s*"', r'\bstd::\w+',
            r'int\s+main\s*\(', r'cout\s*<<', r'cin\s*>>',
            r'\bnullptr\b', r'template\s*<', r'::\w+\(',
            r'\bvector<', r'\bstring\b', r'#define\s+',
        ],
        "weak": [
            r'\bclass\s+\w+\s*[{:]', r'\bvirtual\s+', r'->\w+',
            r'\bconst\s+\w+&', r'\bauto\s+\w+\s*=',
        ],
        "frameworks": {
            "qt": [r'#include\s*<Q', r'Q_OBJECT', r'QWidget', r'signals:'],
            "boost": [r'boost::', r'#include\s*<boost/'],
        },
    },
    "go": {
        "extensions": [".go"],
        "strong": [
            r'^\s*package\s+\w+', r'^\s*func\s+\w+\s*\(',
            r'\bfmt\.\w+', r':=\s*', r'\bgo\s+func',
            r'\bdefer\s+', r'\bgoroutine', r'import\s*\(',
            r'\bchan\s+\w+', r'\bstruct\s*\{',
        ],
        "weak": [
            r'\.go$', r'err\s*!=\s*nil', r'if\s+err',
            r'make\(\w+', r'\binterface\s*\{',
        ],
        "frameworks": {
            "gin": [r'gin\.', r'c\.JSON', r'gin\.Default'],
            "fiber": [r'fiber\.', r'app\.Get'],
        },
    },
    "rust": {
        "extensions": [".rs"],
        "strong": [
            r'^\s*fn\s+\w+', r'\blet\s+mut\s+', r'\bimpl\s+\w+',
            r'\bpub\s+fn\s+', r'println!\s*\(', r'\bmatch\s+\w+',
            r'Option<', r'Result<', r'\buse\s+\w+::',
            r'&self', r'&mut\s+', r'\bmod\s+\w+',
        ],
        "weak": [
            r'\.unwrap\(\)', r'\.expect\(', r'#\[derive',
            r'cargo\.toml', r'\.clone\(\)',
        ],
        "frameworks": {
            "tokio": [r'#\[tokio', r'tokio::'],
            "actix": [r'actix_web', r'HttpServer'],
            "warp": [r'warp::'],
        },
    },
    "sql": {
        "extensions": [".sql"],
        "strong": [
            r'\bSELECT\s+', r'\bFROM\s+', r'\bWHERE\s+',
            r'\bINSERT\s+INTO', r'\bCREATE\s+TABLE',
            r'\bALTER\s+TABLE', r'\bDROP\s+TABLE',
            r'\bJOIN\s+', r'\bGROUP\s+BY', r'\bORDER\s+BY',
            r'\bUPDATE\s+\w+\s+SET',
        ],
        "weak": [
            r'\bDISTINCT\b', r'\bHAVING\b', r'\bLIMIT\b',
            r'\bUNION\b', r'\bCOUNT\(', r'\bSUM\(',
        ],
        "frameworks": {},
    },
    "html": {
        "extensions": [".html", ".htm"],
        "strong": [
            r'<!DOCTYPE\s+html', r'<html', r'<head', r'<body',
            r'<div\b', r'<span\b', r'<script\b', r'<style\b',
            r'<form\b', r'<input\b', r'<button\b',
        ],
        "weak": [r'<p\b', r'<a\b', r'<img\b', r'class="'],
        "frameworks": {},
    },
    "css": {
        "extensions": [".css", ".scss", ".sass", ".less"],
        "strong": [
            r'[\.\#]\w+\s*\{', r'@media\s*\(', r'@import\s+',
            r'display\s*:', r'position\s*:', r'margin\s*:',
            r'padding\s*:', r'background\s*:', r'color\s*:',
            r'font-size\s*:', r'flex\s*:', r'grid\s*:',
        ],
        "weak": [r'!important', r'@keyframes', r':hover', r'::before'],
        "frameworks": {
            "tailwind": [r'@tailwind', r'@apply'],
        },
    },
    "shell": {
        "extensions": [".sh", ".bash", ".zsh"],
        "strong": [
            r'^#!/bin/(bash|sh|zsh)', r'\becho\s+', r'\bgrep\s+',
            r'\bawk\s+', r'\bsed\s+', r'\bchmod\s+',
            r'\$\{\w+\}', r'\$\(\w+', r'\|\s*\w+',
            r'\bif\s+\[', r'\bfi\b', r'\bdone\b',
        ],
        "weak": [r'\bexport\s+', r'\bsource\s+', r'\balias\s+'],
        "frameworks": {},
    },
    "ruby": {
        "extensions": [".rb", ".erb"],
        "strong": [
            r'^\s*def\s+\w+', r'^\s*end\b', r'^\s*class\s+\w+',
            r'puts\s+', r'\brequire\s+', r'\battr_accessor\b',
            r'\bdo\s*\|', r'\.each\s+do', r'\bnil\b',
        ],
        "weak": [r'\.rb$', r'Gemfile', r'\.new\b'],
        "frameworks": {
            "rails": [r'Rails', r'ActiveRecord', r'ApplicationController'],
        },
    },
    "php": {
        "extensions": [".php"],
        "strong": [
            r'<\?php', r'\$\w+\s*=', r'function\s+\w+\s*\(',
            r'\becho\s+', r'->\w+\(', r'::\w+\(',
            r'\bnamespace\s+', r'\buse\s+\w+\\',
        ],
        "weak": [r'\.php$', r'\barray\(', r'\bforeach\s*\('],
        "frameworks": {
            "laravel": [r'Illuminate\\', r'Artisan', r'Eloquent'],
        },
    },
    "swift": {
        "extensions": [".swift"],
        "strong": [
            r'\bfunc\s+\w+\s*\(', r'\bvar\s+\w+\s*:', r'\blet\s+\w+\s*:',
            r'\bguard\s+let', r'\bimport\s+\w+', r'\bstruct\s+\w+',
            r'\bprotocol\s+\w+', r'\benum\s+\w+', r'\boptional\b',
        ],
        "weak": [r'\.swift$', r'@IBOutlet', r'@IBAction'],
        "frameworks": {
            "swiftui": [r'SwiftUI', r'@State', r'@Binding', r'VStack', r'HStack'],
            "uikit": [r'UIKit', r'UIViewController', r'UIView'],
        },
    },
    "kotlin": {
        "extensions": [".kt", ".kts"],
        "strong": [
            r'\bfun\s+\w+\s*\(', r'\bval\s+\w+', r'\bvar\s+\w+',
            r'\bclass\s+\w+', r'\bobject\s+\w+', r'\bwhen\s*\(',
            r'\bdata\s+class', r'\bsealed\s+class', r'\bcompanion\s+object',
        ],
        "weak": [r'\.kt$', r'\bnull\b', r'\?\.'],
        "frameworks": {
            "ktor": [r'ktor', r'routing', r'call\.respond'],
            "android": [r'import android', r'Activity', r'ViewModel'],
        },
    },
}

# Map language â best model tier
LANGUAGE_TIER_MAP: Dict[str, str] = {
    "python": "code",
    "javascript": "code",
    "typescript": "code",
    "java": "code",
    "cpp": "code",
    "go": "code",
    "rust": "code",
    "sql": "balanced",
    "html": "balanced",
    "css": "balanced",
    "shell": "balanced",
    "ruby": "code",
    "php": "code",
    "swift": "code",
    "kotlin": "code",
}

# Map language â preferred providers
LANGUAGE_PROVIDER_MAP: Dict[str, List[str]] = {
    "python": ["gemini", "mistral", "together"],
    "javascript": ["gemini", "mistral", "together"],
    "typescript": ["gemini", "mistral", "together"],
    "java": ["gemini", "together", "groq"],
    "cpp": ["gemini", "together", "groq"],
    "go": ["gemini", "together", "groq"],
    "rust": ["gemini", "together", "groq"],
    "sql": ["gemini", "cohere", "groq"],
}


class LanguageDetector:
    """
    Detect programming language from code text.
    
    Uses pattern matching with confidence scoring.
    Also detects frameworks for more targeted assistance.
    """

    def __init__(self, config=None):
        self.config = config
        self._cache: Dict[int, LanguageDetection] = {}
        self._cache_size = 50

    def detect(self, text: str) -> LanguageDetection:
        """
        Detect the primary programming language in text.
        
        Returns LanguageDetection with language, confidence, and framework.
        """
        if not text or len(text.strip()) < 5:
            return LanguageDetection(language="unknown", confidence=0.0)

        # Cache check (hash of first 500 chars)
        text_hash = hash(text[:500])
        if text_hash in self._cache:
            return self._cache[text_hash]

        scores: Dict[str, float] = {}

        for lang, rules in LANGUAGE_RULES.items():
            score = 0.0

            # Strong indicators (high weight)
            for pattern in rules["strong"]:
                matches = len(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE))
                score += matches * 3.0

            # Weak indicators (low weight)
            for pattern in rules["weak"]:
                matches = len(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE))
                score += matches * 1.0

            if score > 0:
                scores[lang] = score

        if not scores:
            result = LanguageDetection(language="unknown", confidence=0.0)
        else:
            # Normalize scores
            max_score = max(scores.values())
            total_score = sum(scores.values())

            best_lang = max(scores, key=scores.get)
            confidence = min(max_score / max(total_score, 1) + (max_score / 30), 1.0)

            # Detect framework
            framework = self._detect_framework(text, best_lang)

            # Detect if markup
            is_markup = best_lang in ("html", "css", "xml", "markdown")

            result = LanguageDetection(
                language=best_lang,
                confidence=round(confidence, 2),
                framework=framework,
                file_type=LANGUAGE_RULES.get(best_lang, {}).get("extensions", [""])[0],
                is_markup=is_markup,
            )

        # Cache result
        self._cache[text_hash] = result
        if len(self._cache) > self._cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        return result

    def detect_multiple(self, text: str) -> List[LanguageDetection]:
        """Detect all languages present (e.g., HTML + JavaScript + CSS)."""
        results = []
        scores: Dict[str, float] = {}

        for lang, rules in LANGUAGE_RULES.items():
            score = 0.0
            for pattern in rules["strong"]:
                score += len(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)) * 3
            for pattern in rules["weak"]:
                score += len(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE))
            if score >= 3:
                scores[lang] = score

        total = sum(scores.values()) if scores else 1
        for lang, score in sorted(scores.items(), key=lambda x: -x[1]):
            framework = self._detect_framework(text, lang)
            results.append(LanguageDetection(
                language=lang,
                confidence=round(score / total, 2),
                framework=framework,
                file_type=LANGUAGE_RULES.get(lang, {}).get("extensions", [""])[0],
                is_markup=lang in ("html", "css"),
            ))

        return results

    def _detect_framework(self, text: str, language: str) -> str:
        """Detect the framework being used."""
        rules = LANGUAGE_RULES.get(language, {})
        frameworks = rules.get("frameworks", {})

        best_framework = ""
        best_score = 0

        for fw_name, patterns in frameworks.items():
            score = sum(
                1 for p in patterns
                if re.search(p, text, re.MULTILINE | re.IGNORECASE)
            )
            if score > best_score:
                best_score = score
                best_framework = fw_name

        return best_framework if best_score >= 1 else ""

    def get_model_tier(self, language: str) -> str:
        """Get recommended model tier for a language."""
        return LANGUAGE_TIER_MAP.get(language, "balanced")

    def get_preferred_providers(self, language: str) -> List[str]:
        """Get recommended providers for a language."""
        return LANGUAGE_PROVIDER_MAP.get(language, [])

    def get_language_prompt_hint(self, detection: LanguageDetection) -> str:
        """Generate a prompt hint for the detected language."""
        if detection.language == "unknown":
            return ""

        hint = f"Detected language: {detection.language}"
        if detection.framework:
            hint += f" ({detection.framework})"
        if detection.confidence < 0.5:
            hint += " (low confidence)"

        # Language-specific tips
        tips = {
            "python": "Follow PEP 8. Use type hints. Prefer f-strings.",
            "javascript": "Use modern ES6+. Prefer const/let over var.",
            "typescript": "Include proper type annotations. Use strict mode.",
            "java": "Follow Java naming conventions. Handle checked exceptions.",
            "cpp": "Use modern C++ (C++17+). Prefer smart pointers. RAII.",
            "go": "Follow Go idioms. Handle errors explicitly. Use gofmt style.",
            "rust": "Use ownership/borrowing correctly. Handle Result/Option. No unsafe unless needed.",
            "sql": "Use parameterized queries. Avoid SELECT *.",
        }

        if detection.language in tips:
            hint += f"\nStyle guide: {tips[detection.language]}"

        return hint