import sys
import os
import platform

print("=== Python Environment Diagnostic ===")
print(f"Python Version: {sys.version}")
print(f"Executable Path: {sys.executable}")
print(f"Operating System: {platform.system()} {platform.release()}")
print("\n=== Import Search Paths (sys.path) ===")
for p in sys.path:
    print(f"- {p}")

print("\n=== Current Working Directory ===")
print(os.getcwd())

print("\n=== Dependency Check ===")
dependencies = ['cv2', 'numpy', 'supervision', 'ultralytics']
for dep in dependencies:
    try:
        mod = __import__(dep)
        print(f"[SUCCESS] {dep} is available (Version: {getattr(mod, '__version__', 'unknown')})")
    except ImportError:
        print(f"[FAILURE] {dep} is NOT found")

print("\n=== Local Module Check ===")
local_files = ['detector.py', 'pitch_reg.py', 'visualizer.py', 'utils.py']
for f in local_files:
    if os.path.exists(f):
        print(f"[FOUND] {f}")
    else:
        print(f"[MISSING] {f}")
