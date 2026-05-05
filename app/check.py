# check_dependencies.py
# Restaurant Comparison App — ITM352
# Run this first to verify all required packages are installed.

import importlib
import sys

# ── Packages required for this project ───────────────────────────────────────
# Format: "import_name": "pip install name"  (they differ for some packages)
REQUIRED = {
    "flask":            "flask",
    "serpapi":          "google-search-results",
    "pandas":           "pandas",
    "numpy":            "numpy",
    "matplotlib":       "matplotlib",
    "plotly":           "plotly",
    "requests":         "requests",
}

print("=" * 55)
print("   Dependency Check — Restaurant Comparison App")
print("=" * 55)

all_installed = True
missing = []

for import_name, pip_name in REQUIRED.items():
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "version unknown")
        print(f"  ✅ {pip_name:<25} INSTALLED   (version: {version})")
    except ImportError:
        print(f"  ❌ {pip_name:<25} NOT INSTALLED")
        missing.append(pip_name)
        all_installed = False

print("=" * 55)
if all_installed:
    print("  ✅ All packages are installed and ready to go!")
else:
    print("  ⚠️  Some packages are missing.")
    print("     Run the following to install what's missing:")
    print()
    print(f"     pip install {' '.join(missing)}")
print("=" * 55)
print(f"\nPython version: {sys.version}")
