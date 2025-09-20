"""
Setup script for CORD-19 Analysis Project
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_data_file():
    """Check if metadata.csv exists"""
    if os.path.exists("metadata.csv"):
        print("✅ metadata.csv found!")
        return True
    else:
        print("❌ metadata.csv not found!")
        print("\nPlease download the metadata.csv file from:")
        print("https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge")
        print("\nPlace the file in the same directory as this script.")
        return False

def main():
    """Main setup function"""
    print("CORD-19 Analysis Project Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check data file
    if not check_data_file():
        print("\n⚠️  Setup incomplete: Missing data file")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nYou can now run:")
    print("1. python data_analysis.py  (for complete analysis)")
    print("2. streamlit run streamlit_app.py  (for interactive app)")
    
    return True

if __name__ == "__main__":
    main()
