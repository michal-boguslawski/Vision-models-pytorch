import subprocess
import sys

# Function to install pip packages
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Function to run system commands (e.g., apt-get)
def run_system_command(command):
    subprocess.check_call(command, shell=True)

# List of pip packages
pip_packages = [
    "torchmetrics",
    "scikit-image",
    "pycocotools",
    "matplotlib",
    "torch",
    "torchvision",
    "opencv-python",
    "transformers"
]

# Upgrade pip first
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

# Install each package in the list
for package in pip_packages:
    install_package(package)

# Install system package libgl1 (only needed on some systems like Ubuntu)
run_system_command("sudo apt-get install -y libgl1")