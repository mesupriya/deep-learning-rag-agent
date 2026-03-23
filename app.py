import sys
from pathlib import Path

# Add src to the Python path so local imports work seamlessly in HuggingFace
sys.path.append(str(Path(__file__).parent / "src"))

# Import and run the UI main function directly
from rag_agent.ui.app import main

if __name__ == "__main__":
    main()
