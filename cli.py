#!/usr/bin/env python3
"""
CLI Entry Point cho DeepCode-Insight
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

from deepcode_insight.cli.cli import cli

if __name__ == '__main__':
    cli() 