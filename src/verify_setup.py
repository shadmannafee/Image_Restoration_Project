import os
import sys

def check_env():
    print(f"Python Version: {sys.version}")
    print(f"Current Directory: {os.getcwd()}")
    
    # Check for the data folder ----- the one we created now
    if os.path.exists('./data'):
        print("Data folder detected.")
    else:
        print("Data folder missing! Create it in your root directory.")

if __name__ == "__main__":
    check_env()