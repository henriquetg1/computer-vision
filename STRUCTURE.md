# Repository Structure

This document explains the organization of the Computer Vision course repository.

## Directory Tree

```
computer-vision-course/
│
├── README.md                      # Main course overview and syllabus summary
├── SYLLABUS.md                    # Detailed week-by-week syllabus
├── GETTING_STARTED.md             # Environment setup guide
├── RESOURCES.md                   # Learning resources and references
├── STRUCTURE.md                   # This file - repository organization
├── requirements.txt               # Python package dependencies
│
├── lectures/                      # Lecture slides (PDF format)
│   ├── 01.pdf                    # Week 1: Introduction
│   ├── 02.pdf                    # Week 2: Neural Networks Basics
│   ├── 03.pdf                    # Week 3: Color Perception
│   ├── 04.pdf                    # Week 4: Brightness and Contrast
│   ├── 05.pdf                    # Week 5: Advanced Neural Networks
│   ├── 06.pdf                    # Week 6: Image Smoothing
│   ├── 07.pdf                    # Week 7: Discrete Derivatives
│   ├── 08.pdf                    # Week 8: Scale Space
│   ├── 09.pdf                    # Week 9: Corner Detection
│   ├── 10.pdf                    # Week 10: Geometric Transformations
│   ├── 11.pdf                    # Week 11: Deep Neural Networks
│   └── 12.pdf                    # Week 12: Non-Linearity
│
├── labs/                          # Lab assignments and starter code
│   ├── lab01_python_review/
│   │   ├── README.md             # Lab 1 instructions
│   │   ├── task1_load_display.ipynb
│   │   ├── task2_crop_resize.ipynb
│   │   ├── task3_effects.ipynb
│   │   └── starter_code/
│   │
│   ├── lab02_neural_networks/
│   │   ├── README.md             # Lab 2 instructions
│   │   ├── task1_tensors.ipynb
│   │   ├── task2_manual_nn.ipynb
│   │   ├── task3_mnist.ipynb
│   │   └── starter_code/
│   │
│   ├── lab03_color_models/        # Week 3
│   ├── lab04_intensity/           # Week 4
│   ├── lab05_cnns/                # Week 5
│   ├── lab06_filtering/           # Week 6
│   ├── lab07_edges/               # Week 7
│   ├── lab08_pyramids/            # Week 8
│   ├── lab09_features/            # Week 9
│   ├── lab10_geometry/            # Week 10
│   ├── lab11_deep_learning/       # Week 11
│   └── lab12_architectures/       # Week 12
│
├── project/                       # Final project resources
│   ├── README.md                 # Project guidelines and requirements
│   ├── proposal_template.md      # Template for project proposals
│   ├── rubric.md                 # Detailed grading rubric
│   ├── examples/                 # Example project ideas
│   │   ├── example1_plant_disease.md
│   │   ├── example2_document_scanner.md
│   │   └── example3_face_reconstruction.md
│   └── past_projects/            # Showcase of previous student projects
│
├── resources/                     # Additional learning resources
│   ├── datasets/                 # Information about datasets
│   │   ├── README.md
│   │   └── download_scripts/
│   │
│   ├── reference_papers/         # Important papers in CV
│   │   ├── classical_cv/
│   │   └── deep_learning/
│   │
│   ├── tutorials/                # Additional tutorials
│   │   ├── numpy_basics.md
│   │   ├── pytorch_intro.md
│   │   ├── opencv_guide.md
│   │   └── visualization.md
│   │
│   ├── sample_images/            # Test images for labs
│   │   ├── lena.png
│   │   ├── cameraman.png
│   │   └── ...
│   │
│   └── code_examples/            # Reference implementations
│       ├── classical_cv/
│       │   ├── edge_detection.py
│       │   ├── feature_matching.py
│       │   └── ...
│       └── deep_learning/
│           ├── simple_cnn.py
│           ├── transfer_learning.py
│           └── ...
│
├── solutions/                     # Lab solutions (instructor access only)
│   ├── lab01/
│   ├── lab02/
│   └── ...
│
└── utils/                         # Utility scripts
    ├── setup_checker.py          # Verify environment setup
    ├── download_data.py          # Download course datasets
    ├── visualization.py          # Plotting utilities
    └── evaluation.py             # Metrics and evaluation tools
```

## File Descriptions

### Root Level Files

- **README.md:** First file students should read. Contains:
  - Course overview
  - Learning objectives
  - Weekly schedule table
  - Assessment breakdown
  - Getting started instructions

- **SYLLABUS.md:** Detailed syllabus with:
  - Week-by-week breakdown
  - Learning objectives per week
  - Lecture topics
  - Lab exercises
  - Reading assignments

- **GETTING_STARTED.md:** Complete setup guide including:
  - Python installation
  - Virtual environment setup
  - Package installation
  - Testing procedures
  - Troubleshooting

- **RESOURCES.md:** Curated collection of:
  - Textbooks and references
  - Online courses
  - Datasets
  - Software tools
  - Research papers
  - Blogs and communities

- **requirements.txt:** Python dependencies
  - Core packages (PyTorch, OpenCV, NumPy)
  - Optional packages commented out
  - Version constraints where needed

### Lectures Directory

Contains 12 PDF slide decks from your colleague:

1. **01.pdf:** Introduction to Computer Vision
2. **02.pdf:** The Very Basic Basics of Neural Networks
3. **03.pdf:** Color Perception and Color Models
4. **04.pdf:** Brightness and Contrast Adjustment
5. **05.pdf:** Slightly Less Basic Basics of Neural Networks
6. **06.pdf:** Image Smoothing and Convolutional Filters
7. **07.pdf:** Discrete Derivatives as Convolutional Filters
8. **08.pdf:** The Scale Space and Convolutional Layers
9. **09.pdf:** Corner Detection and Keypoint Matching
10. **10.pdf:** Position Models as Geometric Transformations
11. **11.pdf:** The Shallow Basics of Deep Neural Networks
12. **12.pdf:** The Simplicity and Power of Non-Linearity

**Note:** Weeks 13-14 have no lectures (project work and presentations).

### Labs Directory

Each lab follows this structure:
```
labXX_topic_name/
├── README.md              # Detailed instructions
├── taskN_description.ipynb  # Jupyter notebooks for tasks
├── starter_code/          # Starter files for students
│   ├── utils.py
│   └── config.py
├── data/                  # Sample data for the lab (if needed)
└── expected_outputs/      # Example outputs for verification
```

**Lab Topics:**
- **Lab 1:** Python/NumPy review, image I/O
- **Lab 2:** Neural networks with PyTorch
- **Lab 3:** Color space transformations
- **Lab 4:** Histogram processing
- **Lab 5:** CNNs on CIFAR-10
- **Lab 6:** Implementing filters
- **Lab 7:** Edge detection algorithms
- **Lab 8:** Image pyramids
- **Lab 9:** SIFT and feature matching
- **Lab 10:** Homography and image warping
- **Lab 11:** Transfer learning
- **Lab 12:** Architecture experiments

### Project Directory

- **README.md:** Complete project guidelines
  - Scope and requirements
  - Suggested topics
  - Team formation
  - Timeline
  - Deliverables

- **proposal_template.md:** Structured template with:
  - Problem statement
  - Literature review
  - Approach
  - Dataset
  - Evaluation plan
  - Timeline

- **rubric.md:** Detailed grading criteria
  - Code quality
  - Technical depth
  - Results
  - Presentation
  - Report

### Resources Directory

**datasets/:**
- Information about standard datasets
- Download scripts
- Data preparation guides

**reference_papers/:**
- Classic papers (SIFT, HOG, R-CNN, etc.)
- Organized by topic
- Summaries and notes

**tutorials/:**
- Supplementary learning materials
- Step-by-step guides
- Best practices

**sample_images/:**
- Standard test images (Lena, Cameraman, etc.)
- For use in labs and demos

**code_examples/:**
- Reference implementations
- Well-documented examples
- Can be used as starting points

### Solutions Directory

**Access:** Instructor only (not pushed to student repo)

Contains:
- Complete lab solutions
- Multiple approaches where applicable
- Detailed explanations
- Grading notes

### Utils Directory

Helper scripts:
- **setup_checker.py:** Verifies environment
- **download_data.py:** Downloads required datasets
- **visualization.py:** Common plotting functions
- **evaluation.py:** Standard metrics implementations

## Usage Workflows

### For Students

**Week 1:**
```bash
1. git clone <repo-url>
2. cd computer-vision-course
3. Read README.md
4. Follow GETTING_STARTED.md
5. Run python utils/setup_checker.py
6. Review lectures/01.pdf
7. Start labs/lab01_python_review/
```

**Subsequent Weeks:**
```bash
1. git pull  # Get updates
2. Review lectures/XX.pdf
3. Read labs/labXX_topic/README.md
4. Complete lab tasks
5. Submit via course platform
```

**Project Weeks:**
```bash
1. Read project/README.md
2. Form team
3. Fill project/proposal_template.md
4. Develop project
5. Use project/rubric.md for self-assessment
```

### For Instructors

**Setup:**
```bash
1. Clone repository
2. Access solutions/ directory
3. Prepare datasets
4. Configure course platform
```

**Each Week:**
```bash
1. Review lecture slides
2. Test lab assignments
3. Update solutions if needed
4. Grade submissions
5. Provide feedback
```

**Project Management:**
```bash
1. Review proposals
2. Track checkpoint presentations
3. Monitor progress
4. Grade final deliverables using rubric
```

## Maintenance and Updates

### Updating Course Content

**Adding a new lab:**
```bash
1. Create labs/labXX_new_topic/
2. Write README.md with instructions
3. Create Jupyter notebooks
4. Prepare starter code
5. Test thoroughly
6. Update main README.md schedule
```

**Updating slides:**
```bash
1. Edit source slides
2. Export to PDF
3. Replace lectures/XX.pdf
4. Update any lab references if needed
```

**Adding resources:**
```bash
1. Add files to appropriate resources/ subdirectory
2. Update RESOURCES.md with links/descriptions
3. Test all links
```

### Version Control Best Practices

- **Main branch:** Stable, tested content only
- **Development branch:** Work-in-progress updates
- **Release tags:** v2026.1, v2026.2 for each semester

## Student Submission Structure

Expected structure for lab submissions:

```
labXX_studentname/
├── completed_notebooks/
│   ├── task1.ipynb
│   ├── task2.ipynb
│   └── task3.ipynb
├── code/
│   └── helper_functions.py
├── outputs/
│   └── results_images/
├── README.md              # Brief summary
└── reflection.pdf         # Questions answered
```

Expected structure for project submissions:

```
project_teamname/
├── code/
│   ├── src/
│   ├── models/
│   ├── data/
│   ├── configs/
│   └── README.md
├── report/
│   └── final_report.pdf
├── presentation/
│   └── slides.pdf
├── demo/
│   └── demo_video.mp4
└── README.md
```

## Notes for Teaching Assistants

- Solutions in `solutions/` directory
- Use `project/rubric.md` for consistent grading
- Weekly office hours schedule in main README
- Common student issues tracked in GETTING_STARTED.md

## Future Enhancements (Planned)

- [ ] Add autograding scripts for labs
- [ ] Create video lecture supplements
- [ ] Develop interactive web demos
- [ ] Add more example projects
- [ ] Create quiz bank
- [ ] Develop Docker container for consistent environment

---

**Questions about structure?** Post on the course forum or email the instructor.
