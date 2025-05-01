import os
import sys
import subprocess

if __name__ == "__main__":
    # Get the path to the Streamlit script to run
    if len(sys.argv) > 1:
        streamlit_script = sys.argv[1]
    else:
        print("Please provide the path to your Streamlit script as an argument")
        sys.exit(1)
    
    # Additional arguments to pass to streamlit
    additional_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    # Run streamlit with server.runOnSave and server.fileWatcherType disabled
    cmd = [
        "streamlit", "run", 
        streamlit_script,
        "--server.runOnSave=false",
        "--server.fileWatcherType=none"
    ] + additional_args
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
