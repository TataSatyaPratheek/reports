"""
Setup script for Tourism RAG Chatbot.
Handles directory creation and optional dependency installation.
"""
import os
import sys
import argparse
import subprocess
import platform
from typing import List, Dict, Any, Tuple

def create_directory_structure():
    """Create the necessary directory structure for the application."""
    print("Creating directory structure for Tourism RAG Chatbot...")
    
    # Create required directories
    directories = [
        "assets",
        "assets/roles",
        "logs",
        "chroma_vector_db",
        "tests",
        "tests/stress",
        "tests/e2e",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create placeholder files if they don't exist
    placeholder_files = [
        "tests/__init__.py",
        "tests/stress/__init__.py",
        "tests/stress/load_test.py",
        "tests/e2e/__init__.py",
        "tests/e2e/test_app.py",
    ]
    
    for file_path in placeholder_files:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write("# Placeholder file created by setup.py\n")
            print(f"✓ Created placeholder file: {file_path}")
    
    print("Directory structure created successfully!")

def install_dependencies(with_dev: bool = False, with_tourism: bool = False):
    """Install dependencies using PDM or pip."""
    print("Checking package manager...")
    
    # Check if PDM is installed
    pdm_installed = False
    try:
        result = subprocess.run(["pdm", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            pdm_installed = True
            print(f"✓ PDM detected: {result.stdout.strip()}")
    except:
        print("✕ PDM not found. Will try using pip instead.")
    
    if pdm_installed:
        print("Installing dependencies with PDM...")
        commands = [
            ["pdm", "update"],
            ["pdm", "sync"]
        ]
        
        if with_dev:
            commands.append(["pdm", "install", "-G", "dev"])
            
        if with_tourism:
            commands.append(["pdm", "install", "-G", "tourism"])
        
        for cmd in commands:
            try:
                print(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True)
                if result.returncode == 0:
                    print(f"✓ Successfully executed: {' '.join(cmd)}")
                else:
                    print(f"✕ Command failed: {' '.join(cmd)}")
            except Exception as e:
                print(f"✕ Error executing command: {' '.join(cmd)}")
                print(f"  Error details: {str(e)}")
    else:
        print("Installing dependencies with pip...")
        
        # Create requirements files from pyproject.toml
        try:
            if os.path.exists("pyproject.toml"):
                # Extract dependencies from pyproject.toml
                with open("pyproject.toml", "r") as f:
                    content = f.read()
                
                # Parse dependencies section
                main_deps = []
                dev_deps = []
                tourism_deps = []
                
                in_dependencies = False
                in_dev_deps = False
                in_tourism_deps = False
                
                for line in content.splitlines():
                    line = line.strip()
                    
                    if line == "dependencies = [":
                        in_dependencies = True
                        continue
                    elif line.startswith("[project.optional-dependencies]"):
                        in_dependencies = False
                    elif line == "dev = [":
                        in_dev_deps = True
                        continue
                    elif line == "tourism = [":
                        in_tourism_deps = True
                        continue
                    elif line.startswith("]"):
                        in_dependencies = False
                        in_dev_deps = False
                        in_tourism_deps = False
                        continue
                    
                    if in_dependencies and line.startswith('"'):
                        dep = line.strip('",')
                        main_deps.append(dep)
                    elif in_dev_deps and line.startswith('"'):
                        dep = line.strip('",')
                        dev_deps.append(dep)
                    elif in_tourism_deps and line.startswith('"'):
                        dep = line.strip('",')
                        tourism_deps.append(dep)
                
                # Write requirements files
                with open("requirements.txt", "w") as f:
                    f.write("\n".join(main_deps))
                
                if with_dev:
                    with open("requirements-dev.txt", "w") as f:
                        f.write("\n".join(main_deps + dev_deps))
                
                if with_tourism:
                    with open("requirements-tourism.txt", "w") as f:
                        f.write("\n".join(main_deps + tourism_deps))
                
                # Install dependencies
                try:
                    print("Installing main dependencies...")
                    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
                    print("✓ Main dependencies installed successfully")
                    
                    if with_dev:
                        print("Installing development dependencies...")
                        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"], check=True)
                        print("✓ Development dependencies installed successfully")
                    
                    if with_tourism:
                        print("Installing tourism-specific dependencies...")
                        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-tourism.txt"], check=True)
                        print("✓ Tourism-specific dependencies installed successfully")
                except Exception as e:
                    print(f"✕ Error installing dependencies: {str(e)}")
            else:
                print("✕ pyproject.toml not found. Cannot extract dependencies.")
        except Exception as e:
            print(f"✕ Error processing dependencies: {str(e)}")

def setup_spacy_model():
    """Download spaCy model if not already present."""
    print("Checking spaCy model...")
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            print("✓ spaCy model 'en_core_web_sm' already installed")
        except:
            print("✕ spaCy model 'en_core_web_sm' not found. Downloading...")
            result = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            if result.returncode == 0:
                print("✓ spaCy model downloaded successfully")
    except ImportError:
        print("✕ spaCy not installed. Skipping model download.")

def check_ollama():
    """Check if Ollama is installed and suggest download if not."""
    print("Checking Ollama installation...")
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Ollama detected: {result.stdout.strip()}")
            
            # Check available models
            try:
                models_result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                if "llama3.2:latest" in models_result.stdout:
                    print("✓ Recommended model 'llama3.2' is available")
                else:
                    print("ℹ Recommended model 'llama3.2' not found.")
                    print("  You can download it with: ollama pull llama3.2:latest")
            except:
                print("ℹ Could not check Ollama models")
        else:
            print("✕ Ollama command failed")
    except:
        print("✕ Ollama not found. Please install Ollama from https://ollama.ai/download")
        
        # Provide OS-specific installation instructions
        if platform.system() == "Darwin":  # macOS
            print("  macOS installation:")
            print("    curl -fsSL https://ollama.com/install.sh | sh")
        elif platform.system() == "Linux":
            print("  Linux installation:")
            print("    curl -fsSL https://ollama.com/install.sh | sh")
        elif platform.system() == "Windows":
            print("  Windows installation:")
            print("    Download from: https://ollama.ai/download/windows")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Tourism RAG Chatbot")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--tourism", action="store_true", help="Install tourism-specific dependencies")
    parser.add_argument("--all", action="store_true", help="Install all dependencies")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Tourism RAG Chatbot Setup")
    print("=" * 60)
    
    # Create directory structure
    create_directory_structure()
    
    # Install dependencies
    with_dev = args.dev or args.all
    with_tourism = args.tourism or args.all
    install_dependencies(with_dev=with_dev, with_tourism=with_tourism)
    
    # Set up spaCy model
    setup_spacy_model()
    
    # Check Ollama
    check_ollama()
    
    print("\nSetup complete!")
    print("\nTo run the application:")
    print("  - Streamlit interface: streamlit run app.py")
    print("  - Chainlit interface: chainlit run chainlit_app.py")
    print("\nEnjoy your Tourism RAG Chatbot!")

if __name__ == "__main__":
    main()