#!/usr/bin/env python3

import streamlit.web.cli as stcli
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "src/ai_dashboard.py"]
    sys.exit(stcli.main())
