#!/usr/bin/env python
"""
Simple runner script for the Quant Analytics Platform
"""

import subprocess
import sys

if __name__ == "__main__":
    print("Starting Quant Analytics Platform...")
    print("=" * 50)
    print("The application will open in your default browser.")
    print("Press Ctrl+C to stop the application.")
    print("=" * 50)
    
    try:
        # Run streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

