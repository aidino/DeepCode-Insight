#!/usr/bin/env python3
"""
LangGraph Demo - Entry Point
Chạy demo với hai agents truyền message cho nhau.
"""

import sys
import os

# Thêm src vào Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.graph import run_demo

if __name__ == "__main__":
    run_demo() 