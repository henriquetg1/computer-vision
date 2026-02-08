# Final Project Guidelines

## Overview

The final project is an opportunity to apply the computer vision techniques you've learned to a problem of your choice. You'll work in teams of 2-3 students to design, implement, and evaluate a complete computer vision system.

## Timeline

- **Week 5:** Project proposal due
- **Week 9:** Midterm checkpoint presentation
- **Week 13:** Code freeze, final testing
- **Week 14:** Final presentations and deliverables

## Project Scope

Your project should demonstrate mastery of both classical computer vision techniques and modern deep learning approaches. At minimum, your project must:

1. **Include both classical CV and deep learning components**
   - Example: Use SIFT for feature detection + CNN for classification
   - Example: Use edge detection for preprocessing + semantic segmentation network

2. **Solve a real problem with practical applications**
   - Not just a tutorial reproduction
   - Should have potential real-world impact

3. **Include quantitative evaluation**
   - Appropriate metrics for your task
   - Comparison with baselines
   - Analysis of failure cases

## Suggested Project Categories

### 1. Object Detection and Recognition
- Custom object detector for specific domain (e.g., wildlife, defects, medical)
- Real-time recognition system for accessibility
- Document understanding and OCR enhancement

### 2. Image Enhancement and Restoration
- Medical image enhancement
- Old photo restoration
- Super-resolution for specific image types
- HDR imaging from multiple exposures

### 3. Image Segmentation
- Semantic segmentation for specific application
- Interactive segmentation tool
- Background removal for specific scenarios
- Agricultural crop/disease segmentation

### 4. Motion and Tracking
- Real-time object tracking
- Pose estimation application
- Gesture recognition system
- Activity recognition from video

### 5. 3D and Geometry
- 3D reconstruction from images
- Depth estimation
- Panorama and photo stitching
- Visual SLAM

### 6. Creative Applications
- Artistic style transfer with control
- Image generation/manipulation
- Facial editing with specific effects
- Image-based search engine

### 7. Domain-Specific Applications
- Medical imaging analysis
- Agricultural monitoring
- Industrial quality control
- Remote sensing/satellite imagery analysis
- Sports analytics

## Team Formation

- **Team size:** 2-3 students
- Form teams by Week 4
- Individual projects allowed with instructor approval (reduced scope expected)

## Deliverables

### 1. Project Proposal (Week 5) - 5%

**Length:** 2-3 pages

**Contents:**
- **Problem statement:** What problem are you solving? Why is it important?
- **Related work:** What existing solutions are there? What are their limitations?
- **Proposed approach:** What techniques will you use? Why?
- **Dataset:** What data will you use? How will you collect/prepare it?
- **Evaluation plan:** How will you measure success?
- **Timeline:** Milestones for weeks 6-14
- **Team member responsibilities**

**Deliverable:** PDF document via course submission system

---

### 2. Midterm Checkpoint (Week 9) - 5%

**Format:** 5-minute presentation + 3 min Q&A

**Contents:**
- Progress update
- Preliminary results
- Challenges encountered
- Revised plan if needed
- Demo of work-in-progress

**Deliverable:** Presentation slides + working code demo

---

### 3. Final Code Repository (Week 14) - 25%

**Platform:** GitHub (public or private)

**Required Structure:**
```
project-name/
├── README.md              # Project overview, setup instructions, results
├── requirements.txt       # Python dependencies
├── data/
│   └── README.md         # Data sources and preparation instructions
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── utils/
│   └── train.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── experiments/
│   ├── config/           # Configuration files
│   └── results/          # Outputs, metrics, visualizations
├── tests/                # Unit tests (bonus)
└── docs/
    └── architecture.md   # System design documentation
```

**Code Requirements:**
- Clean, well-documented code with docstrings
- Clear README with setup and usage instructions
- Reproducible results (set random seeds)
- Configuration files for experiments
- Trained model weights (or instructions to reproduce)

---

### 4. Final Presentation (Week 14) - 10%

**Format:** 10-minute presentation + 5 min Q&A

**Required Slides:**
1. **Title slide:** Project name, team members
2. **Motivation:** Problem statement and importance
3. **Approach:** Technical methods and architecture
4. **Implementation:** Key design decisions
5. **Results:** Quantitative metrics and qualitative examples
6. **Demo:** Live demonstration (highly recommended)
7. **Analysis:** What worked, what didn't, why?
8. **Conclusions:** Key learnings and future work

**Tips:**
- Practice timing - 10 minutes is strict
- Show demo video if live demo is risky
- Focus on your contributions, not background
- Show interesting failure cases

---

### 5. Final Report (Week 14) - 10%

**Length:** 6-8 pages (excluding references)

**Format:** Conference paper style (IEEE or NeurIPS template recommended)

**Required Sections:**

1. **Abstract** (150-200 words)
   - Problem, approach, key results

2. **Introduction**
   - Motivation and problem statement
   - Contributions and novelty
   - Paper organization

3. **Related Work**
   - Survey of existing methods
   - How your approach differs/improves

4. **Methodology**
   - Detailed technical description
   - Architecture diagrams
   - Algorithm pseudocode if applicable
   - Classical CV components
   - Deep learning components
   - Integration strategy

5. **Experimental Setup**
   - Dataset description and statistics
   - Implementation details
   - Training procedure
   - Evaluation metrics
   - Baseline methods for comparison

6. **Results**
   - Quantitative results with tables/graphs
   - Qualitative results with example images
   - Ablation studies
   - Comparison with baselines
   - Statistical significance if applicable

7. **Discussion**
   - Analysis of results
   - Failure case analysis
   - Limitations of approach
   - Computational requirements

8. **Conclusion**
   - Summary of contributions
   - Future work and improvements

9. **References**
   - Properly cited papers and resources

10. **Appendix** (optional)
    - Additional results
    - Hyperparameter details
    - Code snippets for key algorithms

---

## Evaluation Criteria

### Code Quality (25%)
- **Functionality:** Does it work as intended?
- **Code organization:** Well-structured and modular?
- **Documentation:** Clear comments and README?
- **Reproducibility:** Can results be reproduced?
- **Best practices:** Proper use of version control, configs, etc.

### Technical Depth (30%)
- **Classical CV components:** Appropriate use of traditional techniques
- **Deep learning components:** Proper network design and training
- **Integration:** Effective combination of classical and DL methods
- **Innovation:** Novel aspects or creative solutions
- **Difficulty:** Appropriate challenge level for the course

### Results and Evaluation (20%)
- **Metrics:** Appropriate evaluation methodology
- **Baselines:** Comparison with existing methods
- **Analysis:** Insightful discussion of results
- **Visualization:** Clear presentation of results
- **Ablation studies:** Understanding of what components matter

### Presentation (10%)
- **Clarity:** Easy to follow and understand
- **Demo:** Working demonstration
- **Time management:** Fits within time limit
- **Q&A:** Handles questions well

### Report (10%)
- **Writing quality:** Clear, grammatically correct
- **Technical accuracy:** Correct terminology and concepts
- **Figures/tables:** High-quality visualizations
- **Completeness:** All required sections included
- **References:** Proper citations

### Bonus Points (up to 5%)
- **Open source contribution:** Make your project publicly useful
- **Unit tests:** Comprehensive test coverage
- **Web demo:** Deploy as interactive web application
- **Novel dataset:** Create and share a new dataset
- **Publication quality:** Report approaching publishable standard

## Example Project Ideas

### Beginner-Friendly Projects
1. **Smart Photo Organizer**
   - Face detection + recognition to auto-organize photos
   - Scene classification
   - Duplicate detection using perceptual hashing

2. **Document Scanner App**
   - Perspective correction using homography
   - Adaptive thresholding for binarization
   - OCR integration

3. **Plant Disease Detection**
   - Color-based segmentation for leaves
   - CNN classification of disease types
   - Attention maps to show affected regions

### Intermediate Projects
4. **Real-time Hand Gesture Controller**
   - Hand detection and tracking
   - Keypoint estimation
   - Gesture classification with CNN
   - Integration with system controls

5. **Automatic Panorama Creator**
   - SIFT feature detection and matching
   - RANSAC for homography estimation
   - Multi-band blending for seamless stitching

6. **Medical Image Analyzer**
   - Classical preprocessing (histogram equalization, filtering)
   - Segmentation network (U-Net)
   - Anomaly detection
   - Quantitative measurements

### Advanced Projects
7. **3D Face Reconstruction**
   - Facial landmark detection
   - Depth estimation network
   - 3D mesh generation
   - Texture mapping

8. **Visual SLAM System**
   - Feature tracking across frames
   - Pose estimation
   - Map building
   - Loop closure detection

9. **Deepfake Detection**
   - Face extraction and alignment
   - Temporal inconsistency detection
   - Artifact detection using classical methods
   - Classification network

## Resources

### Datasets
- **General:** ImageNet, COCO, Open Images
- **Faces:** CelebA, WIDER Face, VGGFace2
- **Medical:** ChestX-ray14, ISIC (skin lesions), BraTS
- **Aerial:** SpaceNet, DOTA
- **Custom:** Consider collecting your own!

### Pre-trained Models
- **PyTorch Hub:** torchvision models
- **Hugging Face:** Transformers for vision
- **Model Zoo:** Various framework model collections

### Compute Resources
- **Free GPUs:** Google Colab, Kaggle Kernels
- **Cloud credits:** Inquire about academic cloud credits
- **Insper resources:** Available GPUs for course projects

## Tips for Success

1. **Start early:** Don't wait until week 13!
2. **Iterate quickly:** Get a basic version working, then improve
3. **Version control:** Commit frequently to GitHub
4. **Document as you go:** Don't leave documentation for the end
5. **Test incrementally:** Don't build everything before testing
6. **Communicate:** Update your team and instructor regularly
7. **Manage scope:** Better to do something well than many things poorly
8. **Expect setbacks:** Build buffer time into your schedule
9. **Make it fun:** Choose a project you're excited about!

## Academic Integrity

- All code must be your own or properly attributed
- You may use pre-trained models and libraries (encouraged!)
- Clearly cite any adapted code with source
- Collaboration within team is expected; between teams is not
- Ask if unsure whether something is allowed

## Getting Help

- **Office hours:** Regular times for project consultation
- **Discussion forum:** For general questions
- **Email:** For specific issues
- **Peer feedback:** Teams can give each other feedback

## Frequently Asked Questions

**Q: Can we change our project topic after the proposal?**
A: Minor changes are fine. Major changes require instructor approval.

**Q: What if our initial approach doesn't work?**
A: Document what you tried and why it didn't work. Pivot to a backup approach.

**Q: How much code should we write vs use from libraries?**
A: Use libraries for infrastructure (e.g., PyTorch), but implement key algorithms yourself.

**Q: Can we use ChatGPT or similar tools?**
A: Yes, for learning and debugging, but you must understand all code you submit.

**Q: What if our results aren't as good as we hoped?**
A: Good analysis of why something didn't work can be as valuable as perfect results.

---

**Good luck with your projects! Make something you're proud of!** 🚀📷🤖
