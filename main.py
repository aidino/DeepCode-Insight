#!/usr/bin/env python3
"""
DeepCode-Insight - Main Entry Point
AI-Powered Code Analysis Tool với LangGraph workflow
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for DeepCode-Insight"""
    try:
        from deepcode_insight.core.graph import run_analysis_demo
        
        print("🚀 DeepCode-Insight - AI-Powered Code Analysis Tool")
        print("=" * 60)
        
        # Run the analysis demo
        run_analysis_demo()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 