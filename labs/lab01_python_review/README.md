# Lab 1: Python and Image Processing Basics

**Due Date:** End of Week 1  
**Points:** 100  
**Estimated Time:** 1.5 hours in class + 1 hour homework

## Learning Objectives

By the end of this lab, you will be able to:
- Set up your Python environment for computer vision
- Load, display, and manipulate images using NumPy and PIL/OpenCV
- Understand image representation as arrays
- Perform basic image operations: cropping, resizing, rotating
- Create simple image effects

## Prerequisites

- Python 3.8+ installed
- Basic Python programming knowledge
- Completed environment setup (see main README.md)

## Setup

1. **Create and activate virtual environment:**
```bash
python -m venv cv_env
source cv_env/bin/activate  # Windows: cv_env\Scripts\activate
```

2. **Install requirements:**
```bash
pip install numpy pillow matplotlib opencv-python jupyter
```

3. **Launch Jupyter:**
```bash
jupyter notebook
```

## Part 1: Image Basics (30 min)

### Task 1.1: Load and Display Images

**File:** `task1_load_display.ipynb`

Write code to:
1. Load an image using PIL (Pillow)
2. Convert it to a NumPy array
3. Display the image using matplotlib
4. Print image properties: shape, dtype, min/max values
5. Access and print specific pixel values

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Your code here
```

**Questions to answer:**
- What is the shape of your image array?
- What data type are the pixel values?
- What is the range of pixel values?
- How does the array indexing relate to image coordinates?

### Task 1.2: Understanding Color Channels

Separate and visualize the R, G, and B channels:
1. Split the image into individual color channels
2. Create a 1x3 subplot showing each channel
3. Display each channel as grayscale
4. Create a colored visualization showing which channel is which

**Expected output:** A figure with 4 subplots (original + 3 channels)

### Task 1.3: Grayscale Conversion

Implement grayscale conversion in THREE ways:
1. Using the standard formula: `Gray = 0.299*R + 0.587*G + 0.114*B`
2. Using simple averaging: `Gray = (R + G + B) / 3`
3. Using PIL's built-in convert method

Compare the results visually. Which method gives the most natural-looking result? Why?

## Part 2: Image Manipulations (30 min)

### Task 2.1: Cropping and Resizing

**File:** `task2_crop_resize.ipynb`

1. Load an image
2. Crop the central 50% of the image
3. Resize to 224x224 pixels (common CNN input size)
4. Compare different resize methods (nearest neighbor, bilinear, bicubic)

```python
# Hint for cropping
height, width = img.shape[:2]
start_h, start_w = height // 4, width // 4
end_h, end_w = start_h + height // 2, start_w + width // 2
cropped = img[start_h:end_h, start_w:end_w]
```

### Task 2.2: Rotation and Flipping

Create a function that:
1. Rotates an image by any angle
2. Handles both PIL and NumPy/OpenCV approaches
3. Ensures no clipping occurs

Test with rotations of 45°, 90°, and 180°.

### Task 2.3: Advanced Cropping

Implement a face-centered crop (assuming you have a face image):
1. Use OpenCV's Haar Cascade to detect faces
2. Crop around the detected face with padding
3. Resize to a standard size

```python
import cv2

# Load cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
# Your code here
```

## Part 3: Creative Image Effects (30 min)

### Task 3.1: Photo Mosaic

Create a simple photo mosaic function:
1. Divide the image into NxN tiles
2. Calculate average color of each tile
3. Fill each tile with its average color
4. Create slider to control tile size

**Bonus:** Use actual small images instead of solid colors

### Task 3.2: Sepia Tone Effect

Apply a sepia tone filter using matrix transformation:

```
Sepia matrix:
[0.393, 0.769, 0.189]
[0.349, 0.686, 0.168]
[0.272, 0.534, 0.131]
```

Implement the transformation and handle value clipping.

### Task 3.3: Vignette Effect

Create a vignette (darkened corners) effect:
1. Create a radial gradient mask
2. Apply the mask to darken edges
3. Make it configurable (strength, falloff)

```python
def create_vignette(image, strength=0.5):
    """
    Apply vignette effect to image
    
    Args:
        image: Input image (H, W, C)
        strength: Vignette strength (0-1)
    
    Returns:
        Vignetted image
    """
    # Your implementation
    pass
```

## Homework Assignment (Due Next Week)

### Challenge: Image Collage Generator

Create a program that:
1. Takes a list of image filenames
2. Automatically creates a nice collage layout
3. Resizes images to fit
4. Adds borders and spacing
5. Saves the final result

**Requirements:**
- Must work with any number of images (2-12)
- Should choose layout automatically (2x2, 3x2, 3x3, etc.)
- Images should maintain aspect ratio
- Add creative touches (borders, shadows, titles)

**Deliverable:** 
- Python script `collage_maker.py` that can be run from command line
- Example output images (at least 3 different collages)
- README explaining how to use your script

```bash
# Example usage
python collage_maker.py --images img1.jpg img2.jpg img3.jpg --output collage.jpg
```

## Submission

Submit a ZIP file containing:
1. All completed Jupyter notebooks
2. Your homework script and outputs
3. A brief write-up (PDF) answering:
   - What was the most challenging part?
   - What did you learn about image representation?
   - One interesting thing you discovered while experimenting

**Naming:** `lab1_yourname.zip`

## Grading Rubric

| Component | Points |
|-----------|--------|
| Task 1: Image Basics | 20 |
| Task 2: Manipulations | 20 |
| Task 3: Creative Effects | 20 |
| Homework: Collage Maker | 30 |
| Code Quality & Documentation | 10 |
| **Total** | **100** |

### Code Quality Criteria:
- Clear variable names
- Helpful comments
- Functions with docstrings
- Proper error handling
- Follows PEP 8 style guide

## Resources

### Tutorials:
- [NumPy Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [PIL Documentation](https://pillow.readthedocs.io/)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [Matplotlib Image Tutorial](https://matplotlib.org/stable/tutorials/introductory/images.html)

### Sample Images:
- Use your own photos!
- [Unsplash](https://unsplash.com/) - Free high-quality images
- [Pexels](https://www.pexels.com/) - Free stock photos
- Test images in `/resources/sample_images/`

### Common Pitfalls:
1. **RGB vs BGR:** OpenCV uses BGR, PIL uses RGB!
2. **Integer overflow:** Pixel values must stay in [0, 255]
3. **Data types:** Be careful with uint8 vs float
4. **Indexing:** Remember [height, width] not [width, height]

## Getting Help

- **During lab:** Ask the instructor or TAs
- **Office hours:** Check syllabus for times
- **Discussion forum:** Post questions there
- **Email:** For private concerns only

## Bonus Challenges (Optional)

1. **Performance:** Optimize your code using NumPy vectorization
2. **GUI:** Create a simple GUI for your image effects using tkinter
3. **Batch processing:** Process multiple images at once
4. **EXIF data:** Read and display image metadata

Have fun exploring! 📷✨
