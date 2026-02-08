# Getting Started Guide

Welcome to the Computer Vision course! This guide will help you set up your development environment and get ready for the labs and projects.

## Prerequisites

Before starting, ensure you have:
- Basic Python programming knowledge
- Linear algebra fundamentals (matrices, vectors, transformations)
- A computer with at least 8GB RAM (16GB recommended)
- ~10GB free disk space for datasets and dependencies

## Step 1: Install Python

### Option A: Anaconda (Recommended for Beginners)

1. Download Anaconda from [https://www.anaconda.com/download](https://www.anaconda.com/download)
2. Install with default settings
3. Verify installation:
```bash
conda --version
python --version  # Should be 3.8+
```

### Option B: System Python

**Windows:**
1. Download from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Verify: `python --version`

**macOS:**
```bash
# Using Homebrew
brew install python@3.11
```

**Linux:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

## Step 2: Set Up Virtual Environment

### Using venv (Standard)

```bash
# Create environment
python -m venv cv_env

# Activate environment
# On Windows:
cv_env\Scripts\activate
# On macOS/Linux:
source cv_env/bin/activate

# You should see (cv_env) in your prompt
```

### Using conda

```bash
# Create environment with Python 3.11
conda create -n cv_env python=3.11

# Activate
conda activate cv_env
```

## Step 3: Install Required Packages

### Quick Install (All at Once)

Clone the course repository first:
```bash
git clone https://github.com/insper/computer-vision-2026.git
cd computer-vision-2026
```

Then install dependencies:
```bash
pip install -r requirements.txt
```

This may take 5-10 minutes depending on your internet speed.

### Step-by-Step Install (If Above Fails)

Install packages one by one:

```bash
# Core dependencies
pip install numpy matplotlib pillow

# PyTorch (CPU version - good for starting)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# OpenCV
pip install opencv-python opencv-contrib-python

# Scientific computing
pip install scipy scikit-image scikit-learn pandas

# Jupyter
pip install jupyter notebook ipywidgets

# Additional utilities
pip install tqdm tensorboard
```

### GPU Support (Optional but Recommended)

If you have an NVIDIA GPU:

1. **Check GPU availability:**
```bash
nvidia-smi
```

2. **Install CUDA-enabled PyTorch:**

Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select your configuration.

Example for CUDA 11.8:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Verify GPU access:**
```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Your GPU name
```

## Step 4: Test Your Installation

Create a file `test_setup.py`:

```python
"""
Test script to verify your computer vision environment setup
"""

def test_imports():
    """Test that all required libraries can be imported"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy version:", np.__version__)
    except ImportError:
        print("✗ NumPy not found")
        return False
    
    try:
        import torch
        print("✓ PyTorch version:", torch.__version__)
        print("  CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("  GPU:", torch.cuda.get_device_name(0))
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    try:
        import torchvision
        print("✓ torchvision version:", torchvision.__version__)
    except ImportError:
        print("✗ torchvision not found")
        return False
    
    try:
        import cv2
        print("✓ OpenCV version:", cv2.__version__)
    except ImportError:
        print("✗ OpenCV not found")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL/Pillow available")
    except ImportError:
        print("✗ PIL/Pillow not found")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib version:", matplotlib.__version__)
    except ImportError:
        print("✗ Matplotlib not found")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn version:", sklearn.__version__)
    except ImportError:
        print("✗ scikit-learn not found")
        return False
    
    print("\nAll essential libraries imported successfully!")
    return True


def test_basic_operations():
    """Test basic operations to ensure everything works"""
    print("\nTesting basic operations...")
    
    import numpy as np
    import torch
    from PIL import Image
    
    # NumPy array operation
    arr = np.random.randn(3, 224, 224)
    print("✓ Created NumPy array:", arr.shape)
    
    # PyTorch tensor operation
    tensor = torch.randn(1, 3, 224, 224)
    print("✓ Created PyTorch tensor:", tensor.shape)
    
    # PIL image
    img = Image.new('RGB', (224, 224), color='red')
    print("✓ Created PIL Image:", img.size, img.mode)
    
    # Conversion test
    np_arr = np.array(img)
    print("✓ PIL to NumPy conversion:", np_arr.shape)
    
    torch_tensor = torch.from_numpy(np_arr)
    print("✓ NumPy to PyTorch conversion:", torch_tensor.shape)
    
    print("\nAll basic operations work correctly!")
    return True


def test_gpu():
    """Test GPU if available"""
    import torch
    
    if not torch.cuda.is_available():
        print("\nGPU not available (this is OK for starting out)")
        print("You can use Google Colab for free GPU access")
        return True
    
    print("\nTesting GPU operations...")
    try:
        # Create tensor on GPU
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        
        # Perform operation
        z = torch.matmul(x, y)
        
        print("✓ GPU matrix multiplication successful")
        print(f"  Device: {z.device}")
        return True
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False


def test_jupyter():
    """Check if Jupyter is installed"""
    print("\nTesting Jupyter installation...")
    try:
        import notebook
        print("✓ Jupyter Notebook installed")
        print("  To start: jupyter notebook")
        return True
    except ImportError:
        print("✗ Jupyter not found")
        print("  Install with: pip install jupyter")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Computer Vision Environment Setup Test")
    print("=" * 60)
    
    all_good = True
    
    all_good &= test_imports()
    all_good &= test_basic_operations()
    all_good &= test_gpu()
    all_good &= test_jupyter()
    
    print("\n" + "=" * 60)
    if all_good:
        print("SUCCESS! Your environment is ready for the course! 🎉")
    else:
        print("ISSUES DETECTED - Please fix the errors above")
        print("Check the installation guide or ask for help")
    print("=" * 60)
```

Run the test:
```bash
python test_setup.py
```

## Step 5: Launch Jupyter Notebook

```bash
# Make sure your environment is activated
jupyter notebook
```

This will open Jupyter in your web browser. Navigate to the `labs/` folder to start Lab 1!

## Alternative: Google Colab (Free GPU)

If you don't have a GPU or prefer cloud computing:

1. Go to [https://colab.research.google.com/](https://colab.research.google.com/)
2. Sign in with Google account
3. Click "Upload" and upload any `.ipynb` notebook
4. Enable GPU: Runtime → Change runtime type → GPU

**Advantages:**
- Free GPU access (Tesla T4)
- No installation needed
- Runs in browser

**Disadvantages:**
- Session timeout (12 hours max)
- Can't save files permanently
- Internet connection required

## Troubleshooting

### Common Issues

**Issue:** `pip` command not found
- **Solution:** Use `python -m pip` instead of `pip`

**Issue:** Permission denied when installing packages
- **Solution:** Make sure virtual environment is activated
- **Solution:** Don't use `sudo` with pip

**Issue:** ImportError for packages
- **Solution:** Verify virtual environment is activated
- **Solution:** Reinstall package: `pip install --force-reinstall <package>`

**Issue:** PyTorch not using GPU
- **Solution:** Check CUDA installation: `nvidia-smi`
- **Solution:** Reinstall PyTorch with correct CUDA version

**Issue:** Jupyter kernel not found
- **Solution:** Install ipykernel: `pip install ipykernel`
- **Solution:** Add kernel: `python -m ipykernel install --user --name cv_env`

### Getting Help

1. **Course Forum:** [Link to discussion forum]
2. **Office Hours:** Check syllabus
3. **Documentation:**
   - [PyTorch Installation](https://pytorch.org/get-started/locally/)
   - [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)
4. **Stack Overflow:** Tag with `python`, `pytorch`, `opencv`

## Next Steps

Once your environment is set up:

1. ✅ Read the main [README.md](README.md)
2. ✅ Review the [SYLLABUS.md](SYLLABUS.md)
3. ✅ Check out [RESOURCES.md](RESOURCES.md) for learning materials
4. ✅ Start [Lab 1](labs/lab01_python_review/README.md)

## Recommended IDE Setup

### VS Code (Recommended)

1. Install VS Code from [https://code.visualstudio.com/](https://code.visualstudio.com/)
2. Install Python extension
3. Install Jupyter extension
4. Select your `cv_env` as the Python interpreter (Ctrl/Cmd + Shift + P → "Python: Select Interpreter")

**Useful Extensions:**
- Python (Microsoft)
- Jupyter
- Pylance
- Black Formatter
- GitLens

### PyCharm

1. Download PyCharm Community Edition
2. Open project folder
3. Configure interpreter to use `cv_env`
4. Install Jupyter plugin

### Jupyter Lab (Alternative to Notebook)

More modern interface:
```bash
pip install jupyterlab
jupyter lab
```

## Dataset Download (Optional Now, Required Later)

You'll need these datasets for labs and projects:

```bash
# Create data directory
mkdir -p data

# MNIST (automatic download via torchvision in code)

# Sample images for testing
# Download from course repository or use your own
```

Most datasets will be automatically downloaded when needed via torchvision or other libraries.

## Quick Reference Card

**Activate environment:**
```bash
# venv
source cv_env/bin/activate  # Mac/Linux
cv_env\Scripts\activate     # Windows

# conda
conda activate cv_env
```

**Install package:**
```bash
pip install package_name
```

**Update package:**
```bash
pip install --upgrade package_name
```

**List installed packages:**
```bash
pip list
```

**Launch Jupyter:**
```bash
jupyter notebook
```

**Deactivate environment:**
```bash
deactivate  # venv
conda deactivate  # conda
```

## Tips for Success

1. **Use version control:** Learn basic git commands
2. **Save your work often:** Jupyter auto-saves, but export notebooks too
3. **Experiment:** Don't just run code - modify and try variations
4. **Read errors carefully:** Error messages are your friend
5. **Use print statements:** Debug by printing intermediate values
6. **Visualize:** Always look at your images and data
7. **Ask for help early:** Don't struggle for hours alone
8. **Back up your work:** Use GitHub or cloud storage

## Additional Resources

- **Python Refresher:** [https://docs.python.org/3/tutorial/](https://docs.python.org/3/tutorial/)
- **NumPy Tutorial:** [https://numpy.org/doc/stable/user/quickstart.html](https://numpy.org/doc/stable/user/quickstart.html)
- **PyTorch Tutorials:** [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- **Jupyter Guide:** [https://jupyter-notebook.readthedocs.io/](https://jupyter-notebook.readthedocs.io/)

---

**You're all set! Good luck with the course!** 🚀📷🤖

If you encounter any issues not covered here, please post on the course forum or attend office hours.
