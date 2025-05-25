#!/usr/bin/env python3
"""
DeepCode-Insight - Main Entry Point
AI-Powered Code Analysis Tool với LangGraph workflow
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

from deepcode_insight.core.graph import run_analysis_demo

if __name__ == "__main__":
    print("🚀 DeepCode-Insight - AI-Powered Code Analysis Tool")
    print("=" * 60)
    run_analysis_demo() 