import os
import sys

PROJECT_ROOT = "/opt/spotify-brain"


def enter_project():
    src_path = os.path.join(PROJECT_ROOT, "src")
    for p in (PROJECT_ROOT, src_path):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(PROJECT_ROOT)
