# Detailed Weekly Syllabus

## Week 1: Introduction to Computer Vision

**Learning Objectives:**
- Understand what computer vision is and its applications
- Learn basic image representation (pixels, channels, resolution)
- Set up development environment

**Lecture (~30 min):**
- What is vision? Human vs computer vision
- Applications: face recognition, autonomous vehicles, medical imaging, AR/VR
- Image basics: pixels, coordinates, color channels
- Digital image representation

**Lab (1.5 hours):**
- Python environment setup
- NumPy review: arrays, indexing, broadcasting
- Loading and displaying images with PIL/OpenCV/matplotlib
- Basic image manipulations: crop, resize, rotate
- Exercise: Create a photo mosaic

**Resources:**
- Gonzalez & Woods: Chapter 1, 2
- MIT Vision: Chapter 1

---

## Week 2: The Very Basic Basics of Neural Networks

**Learning Objectives:**
- Understand perceptrons and feedforward networks
- Learn backpropagation algorithm
- Introduction to PyTorch

**Lecture (~30 min):**
- From biological neurons to artificial neurons
- Perceptron model and activation functions
- Forward propagation
- Loss functions and gradient descent
- Backpropagation intuition

**Lab (1.5 hours):**
- PyTorch basics: tensors, autograd
- Implement a simple 2-layer network from scratch
- Train on MNIST digits dataset
- Visualize decision boundaries
- Exercise: Build a network to classify simple shapes

**Resources:**
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- Neural Networks from Scratch (online resource)

---

## Week 3: Color Perception and Color Models

**Learning Objectives:**
- Understand human color perception
- Learn different color spaces (RGB, HSV, LAB)
- Apply color transformations

**Lecture (~30 min):**
- Human visual system and color perception
- RGB color model
- HSV/HSL color models  
- Color space conversions
- Applications: skin detection, object tracking by color

**Lab (1.5 hours):**
- Convert between color spaces
- Histogram analysis in different color spaces
- Color-based segmentation
- White balance correction
- Exercise: Build a green screen (chroma key) effect

**Resources:**
- Gonzalez & Woods: Chapter 6
- MIT Vision: Chapter 4

---

## Week 4: Brightness and Contrast Adjustment

**Learning Objectives:**
- Master intensity transformations
- Understand histogram processing
- Apply adaptive techniques

**Lecture (~30 min):**
- Point operations: negative, log, power-law
- Histogram equalization
- Histogram matching
- Adaptive histogram equalization (CLAHE)
- Gamma correction

**Lab (1.5 hours):**
- Implement histogram equalization from scratch
- Apply CLAHE to medical images
- Automatic brightness/contrast adjustment
- HDR image tone mapping
- Exercise: Enhance low-light photographs

**Resources:**
- Gonzalez & Woods: Chapter 3
- MIT Vision: Chapter 5

---

## Week 5: Slightly Less Basic Basics of Neural Networks

**Learning Objectives:**
- Understand Convolutional Neural Networks (CNNs)
- Learn about activation functions, pooling, dropout
- Master training techniques

**Lecture (~30 min):**
- Limitations of fully-connected networks for images
- Convolutional layers: filters, stride, padding
- Pooling layers (max, average)
- CNN architectures: LeNet, AlexNet overview
- Regularization: dropout, batch normalization
- Optimization: SGD, Adam, learning rate scheduling

**Lab (1.5 hours):**
- Build a CNN in PyTorch for CIFAR-10
- Experiment with different architectures
- Visualize learned filters
- Data augmentation techniques
- Exercise: Fine-tune hyperparameters for best accuracy

**Assignment:** **Project Proposal Due**

**Resources:**
- CS231n Convolutional Neural Networks
- PyTorch CNN tutorial

---

## Week 6: Image Smoothing and Convolutional Filters

**Learning Objectives:**
- Master convolution operation
- Understand different smoothing filters
- Learn noise reduction techniques

**Lecture (~30 min):**
- Convolution in 1D and 2D
- Linear filters: box filter, Gaussian filter
- Non-linear filters: median filter, bilateral filter
- Separable filters for efficiency
- Connection to CNN convolutional layers

**Lab (1.5 hours):**
- Implement convolution from scratch
- Apply various smoothing filters
- Denoise images with different noise types
- Compare filter performance
- Exercise: Build a real-time blur/sharpen tool

**Project Work:** Students work on data collection and preprocessing

**Resources:**
- Gonzalez & Woods: Chapter 3
- MIT Vision: Chapter 6

---

## Week 7: Discrete Derivatives as Convolutional Filters

**Learning Objectives:**
- Understand image gradients
- Learn edge detection algorithms
- Master derivative filters

**Lecture (~30 min):**
- First-order derivatives: Sobel, Prewitt, Roberts
- Second-order derivatives: Laplacian
- Edge detection: Canny edge detector
- Gradient magnitude and direction
- Applications: edge-based segmentation

**Lab (1.5 hours):**
- Implement Sobel filter from scratch
- Full Canny edge detection implementation
- Gradient-based image sharpening
- Line detection with Hough transform
- Exercise: Create artistic edge-based effects

**Resources:**
- Gonzalez & Woods: Chapter 10
- MIT Vision: Chapter 7

---

## Week 8: The Scale Space and Convolutional Layers

**Learning Objectives:**
- Understand scale-space theory
- Learn image pyramids
- Connect to deep network architectures

**Lecture (~30 min):**
- Scale-space representation
- Gaussian pyramids and Laplacian pyramids
- Scale-invariant features motivation
- Multi-scale processing
- Connection to CNN pooling and strided convolutions

**Lab (1.5 hours):**
- Build Gaussian and Laplacian pyramids
- Multi-scale edge detection
- Image blending with pyramids
- Panorama stitching introduction
- Exercise: Create a seamless image blend

**Resources:**
- MIT Vision: Chapter 8
- Szeliski Computer Vision: Chapter 3

---

## Week 9: Corner Detection and Keypoint Matching

**Learning Objectives:**
- Understand interest point detection
- Learn feature descriptors (SIFT, ORB)
- Master feature matching techniques

**Lecture (~30 min):**
- Harris corner detector
- Scale-invariant feature transform (SIFT)
- ORB and other modern descriptors
- Feature matching: brute force, FLANN
- Applications: object recognition, tracking, SLAM

**Lab (1.5 hours):**
- Implement Harris corner detector
- Use SIFT/ORB for feature matching
- Build a simple object recognition system
- Robust matching with RANSAC preview
- Exercise: Create a simple AR marker tracker

**Assignment:** **Midterm Project Checkpoint Due**

**Resources:**
- Szeliski: Chapter 4
- OpenCV Feature Detection tutorials

---

## Week 10: Position Models as Geometric Transformations

**Learning Objectives:**
- Master 2D transformations: affine, homography
- Learn RANSAC for robust estimation
- Apply to image alignment

**Lecture (~30 min):**
- Translation, rotation, scaling, shearing
- Affine transformations (6 DOF)
- Projective transformations / Homography (8 DOF)
- RANSAC algorithm for outlier rejection
- Camera calibration basics

**Lab (1.5 hours):**
- Estimate affine transform from point correspondences
- Implement RANSAC
- Image rectification and warping
- Panorama stitching pipeline
- Exercise: Build an automatic panorama stitcher

**Project Work:** Students refine their implementations

**Resources:**
- Szeliski: Chapter 2, 6
- Multiple View Geometry (Hartley & Zisserman)

---

## Week 11: The Shallow Basics of Deep Neural Networks

**Learning Objectives:**
- Understand modern deep architectures
- Learn transfer learning
- Master advanced training techniques

**Lecture (~30 min):**
- ResNet and skip connections
- Batch normalization deep dive
- Transfer learning and fine-tuning
- Pre-trained models: ImageNet, COCO
- Modern architectures: EfficientNet, Vision Transformers overview

**Lab (1.5 hours):**
- Load and use pre-trained ResNet
- Fine-tune on custom dataset
- Feature extraction with pre-trained networks
- Compare training from scratch vs transfer learning
- Exercise: Build a custom classifier with transfer learning

**Resources:**
- Deep Learning Book (Goodfellow et al.)
- PyTorch Transfer Learning tutorial

---

## Week 12: The Simplicity and Power of Non-Linearity

**Learning Objectives:**
- Understand activation functions deeply
- Learn advanced architectural patterns
- Explore attention mechanisms

**Lecture (~30 min):**
- Activation functions: ReLU, Leaky ReLU, GELU, Swish
- Why depth matters: universal approximation theorem
- Attention mechanisms basics
- Modern tricks: StochasticDepth, MixUp
- Architecture search and AutoML concepts

**Lab (1.5 hours):**
- Experiment with different activation functions
- Implement attention mechanism
- Build and compare different architectures
- Ablation studies
- Exercise: Design and test a novel architecture variant

**Project Work:** Students finalize implementations and testing

**Resources:**
- Attention Is All You Need (Transformer paper)
- Recent architecture papers

---

## Week 13: Final Project Work

**NO LECTURE - FULL LAB TIME (2 hours)**

**Activities:**
- Students work on final projects
- Instructor provides one-on-one consultations
- Debug sessions
- Performance optimization
- Presentation preparation

**Deliverables:**
- Code freeze for final project
- Prepare presentation slides
- Draft final report

---

## Week 14: Final Presentations and Wrap-up

**Format:** Student Presentations

**Schedule:**
- Each team presents for 10 minutes + 5 min Q&A
- Demonstrate working system
- Show results and analysis
- Discuss challenges and solutions

**Final Deliverables Due:**
- Code repository (GitHub)
- Final report (PDF)
- Presentation slides
- Demo video (optional but recommended)

**Instructor Wrap-up (30 min):**
- Course review and key takeaways
- Future directions in computer vision
- Resources for continued learning
- Career opportunities in CV/ML

---

## Topics for Self-Study (Time Permitting)

- Object detection (YOLO, Faster R-CNN)
- Semantic and instance segmentation
- Optical flow and video analysis
- 3D vision and depth estimation
- Generative models (GANs, Diffusion models)
- Vision-language models (CLIP, etc.)

---

*This syllabus is subject to minor adjustments based on class progress and interests.*
