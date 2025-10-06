# app.py - The main entry point for the PaperSage Gradio Dashboard

import sys
import os
import time

# --- Import Path Setup ---
# 1. Get the project root directory (where app.py is located).
project_root = os.path.abspath(os.path.dirname(__file__))

# 2. Add the project root to the system path. This allows Python to recognize
#    'papersage' as a top-level package for absolute imports.
if project_root not in sys.path:
    sys.path.append(project_root)


def main():
    """
    Launches the Gradio application by importing the 'app' object
    from papersage.scripts.dashboard.
    """
    try:
        # Import the Gradio 'app' object directly from the module.
        # This will execute the setup code inside dashboard.py before launching.
        print("Attempting to import Gradio application from 'papersage.scripts.dashboard'...")
        from scripts.dashboard import app

        print("Import successful. Launching Gradio application...")

        # Call the .launch() method on the imported Gradio application instance
        app.launch()

    except ImportError as e:
        print("\n" + "=" * 70)
        print("FATAL ERROR: Could not import the dashboard module.")
        print(f"Details: {e}")
        print("Please ensure the file structure is correct:")
        print("  - ./app.py")
        print("  - ./papersage/scripts/dashboard.py")
        print("  - Necessary files (like 'modules' and '.env') are accessible from app.py.")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"An unexpected error occurred during application startup: {e}")


if __name__ == "__main__":
    main()
