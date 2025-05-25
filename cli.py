#!/usr/bin/env python3
"""
CLI Entry Point cho DeepCode-Insight
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main CLI entry point"""
    try:
        from deepcode_insight.cli.cli import cli
        cli()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running CLI: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 