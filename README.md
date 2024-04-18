## Environment Setup

This section guides you through setting up the development environment necessary to run this project. Follow these steps to ensure that your setup is correct.

### 1. Install CMake

`apriltag` requires CMake to build some of its components. Install CMake using the appropriate command for your operating system:

#### For Ubuntu/Debian
```bash
sudo apt update
sudo apt install cmake
```

#### For Fedora/RHEL/CentOS
```bash
sudo dnf install cmake
```

#### For macOS
```bash
brew install cmake
```

### 2. Set Up a Conda Environment

Creating a Conda environment is recommended to manage the project's dependencies efficiently:

```bash
conda create --name myenv python=3.8  # You can change `myenv` to any name you prefer
conda activate myenv
```

### 3. Install Required Python Packages

Install all the required Python packages using Conda or pip. If you are using Conda, run:

```bash
pip install -r requirements.txt
```


### 4. Verify Installation

Ensure that all packages are installed correctly by running the following command:

```bash
python -c "import cv2, numpy, apriltag; print('All packages installed correctly!')"
```

You should see "All packages installed correctly!" printed if the installation was successful.