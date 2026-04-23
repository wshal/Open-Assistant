import os
import glob
import sys

# Common Windows-1252 to UTF-8 mojibake patterns
MOJIBAKE_MAP = {
    "—": "—",  # em dash
    "’": "’",  # right single quote
    "–": "–",  # en dash
    "“": "“",  # left double quote
    "”": "”",  # right double quote (with space)
    "": "",     # emoji artifact prefix
    "…": "…",  # ellipsis
}

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fix_mode = "--fix" in sys.argv
    found_issues = False

    print(f"Scanning for mojibake in {root_dir}...")
    
    for ext in ["*.py", "*.md", "*.yaml", "*.json"]:
        for filepath in glob.glob(os.path.join(root_dir, "**", ext), recursive=True):
            if "venv" in filepath or "scratch" in filepath or ".git" in filepath:
                continue
                
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                has_mojibake = any(m in content for m in MOJIBAKE_MAP)
                
                if has_mojibake:
                    found_issues = True
                    print("Mojibake found in: " + filepath.encode('ascii', 'ignore').decode())
                    
                    if fix_mode:
                        for bad, good in MOJIBAKE_MAP.items():
                            content = content.replace(bad, good)
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(content)
                        print("  -> Fixed.")
            except UnicodeDecodeError:
                found_issues = True
                print(f"❌ Encoding error (not UTF-8): {os.path.relpath(filepath, root_dir)}")
            except Exception as e:
                pass
                
    if not found_issues:
        print("✅ No mojibake or encoding issues found. All clear!")
    else:
        if not fix_mode:
            print("\nRun with --fix to automatically correct known mojibake.")
        sys.exit(1 if found_issues and not fix_mode else 0)

if __name__ == "__main__":
    main()
