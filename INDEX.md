# Computer Vision Course - Quick Index

**Welcome!** This is your quick navigation guide to the course repository.

## 🚀 Getting Started (START HERE!)

1. **[README.md](README.md)** - Course overview and schedule
2. **[GETTING_STARTED.md](GETTING_STARTED.md)** - Environment setup guide
3. **[SYLLABUS.md](SYLLABUS.md)** - Detailed weekly syllabus

## 📚 Core Course Materials

### Lectures (Week by Week)
- **Week 1:** [Introduction](lectures/01.pdf)
- **Week 2:** [Neural Networks Basics](lectures/02.pdf)
- **Week 3:** [Color Perception](lectures/03.pdf)
- **Week 4:** [Brightness and Contrast](lectures/04.pdf)
- **Week 5:** [Advanced Neural Networks](lectures/05.pdf)
- **Week 6:** [Image Smoothing](lectures/06.pdf)
- **Week 7:** [Discrete Derivatives](lectures/07.pdf)
- **Week 8:** [Scale Space](lectures/08.pdf)
- **Week 9:** [Corner Detection](lectures/09.pdf)
- **Week 10:** [Geometric Transformations](lectures/10.pdf)
- **Week 11:** [Deep Neural Networks](lectures/11.pdf)
- **Week 12:** [Non-Linearity](lectures/12.pdf)

### Lab Assignments
- **Lab 1:** [Python & Image Basics](labs/lab01_python_review/README.md)
- **Lab 2:** [Neural Networks with PyTorch](labs/lab02_neural_networks/README.md)
- **Lab 3-12:** See `labs/` directory for remaining assignments

### Final Project
- **[Project Guidelines](project/README.md)** - Requirements and timeline
- **[Proposal Template](project/proposal_template.md)** - Use this for your proposal
- **[Grading Rubric](project/rubric.md)** - How you'll be graded

## 📖 Additional Resources

- **[RESOURCES.md](RESOURCES.md)** - Textbooks, datasets, tools, papers
- **[STRUCTURE.md](STRUCTURE.md)** - Repository organization guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute improvements

## 🔧 Quick Commands

### First Time Setup
```bash
# Clone repository
git clone <repo-url>
cd computer-vision-course

# Create environment
python -m venv cv_env
source cv_env/bin/activate  # Windows: cv_env\Scripts\activate

# Install packages
pip install -r requirements.txt

# Test setup
python test_setup.py  # (create this from GETTING_STARTED.md)

# Start Jupyter
jupyter notebook
```

### Weekly Workflow
```bash
# Update repository
git pull

# Activate environment
source cv_env/bin/activate

# Review this week's lecture
# Open lectures/XX.pdf

# Start this week's lab
cd labs/labXX_topic
jupyter notebook
```

## 📅 Quick Schedule Reference

| Week | Topic | Lab | Deliverable |
|------|-------|-----|-------------|
| 1 | Introduction | Python Review | - |
| 2 | Neural Networks | PyTorch Basics | - |
| 3 | Color Models | Color Transforms | - |
| 4 | Brightness/Contrast | Histogram Processing | - |
| 5 | CNNs | CIFAR-10 Training | **Project Proposal** |
| 6 | Filtering | Custom Filters | - |
| 7 | Edge Detection | Canny Implementation | - |
| 8 | Scale Space | Image Pyramids | - |
| 9 | Features | SIFT Matching | **Midterm Checkpoint** |
| 10 | Geometry | Image Warping | - |
| 11 | Deep Learning | Transfer Learning | - |
| 12 | Architectures | Model Design | - |
| 13 | Project Work | No Lab | Code Freeze |
| 14 | Presentations | No Lab | **Final Deliverables** |

## 🎯 Key Deadlines

- **Week 5:** Project Proposal Due
- **Week 9:** Midterm Project Checkpoint
- **Week 13:** Final Project Code Freeze
- **Week 14:** Final Presentations & Report

## 💡 Quick Tips

**Environment Issues?**
→ See [GETTING_STARTED.md](GETTING_STARTED.md#troubleshooting)

**Need a Dataset?**
→ See [RESOURCES.md](RESOURCES.md#datasets)

**Stuck on a Lab?**
→ Check the lab's README first, then office hours

**Want to Contribute?**
→ Read [CONTRIBUTING.md](CONTRIBUTING.md)

## 📞 Getting Help

1. **Lab README files** - Check the specific lab's instructions
2. **Course forum** - [Link TBD]
3. **Office hours** - Check syllabus for times
4. **Email instructor** - For private matters only

## 🌟 Pro Tips

- ✅ Read the README for each lab BEFORE starting
- ✅ Commit your work frequently to git
- ✅ Start the project early (Week 5!)
- ✅ Visualize everything - always look at your images
- ✅ Ask questions early and often
- ✅ Experiment beyond the requirements
- ✅ Use Google Colab for free GPU access
- ✅ Collaborate (but don't copy)

## 🗂️ Directory Quick Reference

```
computer-vision-course/
├── README.md              ← Start here
├── GETTING_STARTED.md     ← Setup guide
├── SYLLABUS.md           ← Week by week details
├── RESOURCES.md          ← Learning materials
├── lectures/             ← Lecture slides (PDF)
├── labs/                 ← Weekly assignments
├── project/              ← Final project info
└── requirements.txt      ← Python packages
```

## 🎓 Learning Path

**Absolute Beginners:**
1. Python review → Lab 1
2. PyTorch tutorials → Lab 2
3. CS231n lectures (online)
4. Start simple, build up

**Some Experience:**
1. Skim Python review
2. Dive into PyTorch → Lab 2
3. Focus on computer vision specifics
4. Take on challenging optional tasks

**Advanced:**
1. Skip to CNNs (Week 5)
2. Implement papers from scratch
3. Contribute to course materials
4. Aim for publication-quality project

## 📊 Assessment Breakdown

- **Labs:** 30% (12 labs)
- **Midterm Checkpoint:** 20%
- **Final Project:** 50%
  - Proposal: 5%
  - Code: 25%
  - Presentation: 10%
  - Report: 10%

## 🔗 External Links (Bookmarks)

- **PyTorch Docs:** https://pytorch.org/docs/
- **OpenCV Tutorials:** https://docs.opencv.org/
- **CS231n:** http://cs231n.stanford.edu/
- **Papers with Code:** https://paperswithcode.com/
- **Google Colab:** https://colab.research.google.com/

## ⚡ Quick Answers to Common Questions

**Q: Can I use libraries like scikit-image?**
A: Yes! Use whatever makes sense for the task.

**Q: Do I need a GPU?**
A: Not required, but helpful. Use Google Colab for free GPU.

**Q: Can I work ahead?**
A: Yes! All materials are available from day 1.

**Q: What if I'm behind?**
A: Talk to instructor ASAP. Don't wait until it's too late.

**Q: Can I do an individual project?**
A: Yes, with instructor approval. Reduced scope expected.

**Q: Where are the datasets?**
A: Most download automatically. See RESOURCES.md for others.

---

**Ready to start? Go to [GETTING_STARTED.md](GETTING_STARTED.md)!** 🚀

*Last updated: February 2026*
