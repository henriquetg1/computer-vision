# Contributing to the Course

We welcome contributions from students to improve course materials! This document explains how you can contribute.

## Ways to Contribute

### 1. Report Issues

Found a bug or typo? Please report it!

**How to report:**
1. Check if issue already exists
2. Open a new issue with:
   - Clear title
   - Detailed description
   - Steps to reproduce (if applicable)
   - Your environment details

**Examples of good issues:**
- "Typo in Lab 2 README, line 45"
- "Broken link to dataset in RESOURCES.md"
- "Code in lab03_task1.ipynb throws error on line 23"
- "Unclear instructions in Project Guidelines section 4.2"

### 2. Fix Typos and Small Errors

Help us keep the materials polished!

**Process:**
1. Fork the repository
2. Create a branch: `git checkout -b fix/typo-lab2`
3. Make your changes
4. Commit: `git commit -m "Fix typo in Lab 2 README"`
5. Push: `git push origin fix/typo-lab2`
6. Open a Pull Request

### 3. Improve Documentation

Make instructions clearer for future students!

**What to improve:**
- Clarify confusing explanations
- Add helpful examples
- Expand troubleshooting sections
- Improve code comments

**Process:**
Same as fixing typos, but use branch name like `docs/improve-lab1-instructions`

### 4. Share Resources

Found a great tutorial or dataset?

**How to share:**
1. Add to appropriate section in RESOURCES.md
2. Include:
   - Clear description
   - Link
   - Why it's helpful
   - Appropriate category
3. Submit PR with branch name like `resources/add-pytorch-tutorial`

### 5. Contribute Code Examples

Help fellow students with example implementations!

**Guidelines:**
- Well-commented code
- Follows course style
- Includes documentation
- Tests on common cases
- Add to `resources/code_examples/`

**Example contributions:**
- Alternative implementation of an algorithm
- Visualization helper functions
- Data preprocessing utilities
- Evaluation metrics

### 6. Improve Lab Assignments (Advanced)

Suggest improvements to existing labs!

**What you can propose:**
- Additional exercises
- Better explanations
- More challenging optional tasks
- Real-world applications

**Process:**
1. Discuss idea with instructor first (open issue)
2. Get approval
3. Create detailed proposal
4. Implement and test thoroughly
5. Submit PR with comprehensive description

## Contribution Guidelines

### Code Style

**Python:**
- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and small

```python
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union for two bounding boxes
    
    Args:
        box1: (x1, y1, x2, y2) coordinates
        box2: (x1, y1, x2, y2) coordinates
    
    Returns:
        float: IoU score between 0 and 1
    """
    # Implementation here
    pass
```

**Jupyter Notebooks:**
- Clear markdown explanations
- One concept per cell when possible
- Include expected outputs in comments
- Add visualization for results

### Documentation Style

**Markdown:**
- Use clear headers (##, ###)
- Include code examples in code blocks
- Add images/diagrams when helpful
- Link to relevant resources

**Comments:**
- Explain *why*, not just *what*
- Keep comments up-to-date with code
- Use TODO for incomplete parts

### Commit Messages

Follow this format:

```
<type>: <short description>

<optional longer description>
```

**Types:**
- `fix`: Bug fixes
- `docs`: Documentation changes
- `feat`: New features
- `refactor`: Code improvements
- `test`: Adding tests
- `style`: Formatting changes

**Examples:**
```
fix: Correct tensor shape in Lab 2 task 3

The expected shape was (batch, 28, 28) but should be (batch, 1, 28, 28)
for compatibility with PyTorch Conv2d layers.
```

```
docs: Add troubleshooting for CUDA installation

Added section on common CUDA installation issues and their solutions
based on student forum questions.
```

## Pull Request Process

### Before Submitting

- [ ] Test your changes thoroughly
- [ ] Update relevant documentation
- [ ] Follow code style guidelines
- [ ] Check for typos and formatting
- [ ] Ensure no sensitive information included

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] Documentation improvement
- [ ] New feature/example
- [ ] Resource addition

## Testing Done
How you tested these changes

## Related Issues
Closes #123 (if applicable)

## Screenshots (if applicable)
Add screenshots for visual changes
```

### Review Process

1. Submit PR
2. Instructor/TA reviews
3. Address feedback if requested
4. Changes approved
5. Merged into main branch

**Timeline:**
- Simple fixes: 1-3 days
- Documentation: 3-7 days  
- Code contributions: 1-2 weeks

## Recognition

Contributors will be acknowledged in:
- Course CONTRIBUTORS.md file
- End-of-semester recognition
- GitHub contributor stats

Top contributors may receive:
- Extra credit (up to 2% of final grade)
- Letter of recommendation
- Featured in course materials

## What NOT to Contribute

❌ **Do not contribute:**
- Solutions to current assignments
- Exam answers or hints
- Copyrighted materials without permission
- Low-effort changes (e.g., changing single word)
- Personal opinions as facts
- Unverified information

## Getting Help

**Have questions about contributing?**

1. **Read existing issues/PRs first**
2. **Search documentation**
3. **Ask on course forum**
4. **Email instructor** for sensitive matters

## Community Guidelines

### Be Respectful
- Constructive criticism only
- Acknowledge others' work
- Be patient with review process
- Help other contributors

### Be Clear
- Write clear issue descriptions
- Explain your reasoning
- Provide context
- Include examples

### Be Professional
- Appropriate language
- No spam or off-topic content
- Respect intellectual property
- Follow academic integrity

## Special Contribution Opportunities

### Document Your Project
After completing your final project:
- Share your approach in `project/examples/`
- Add to `project/past_projects/` showcase
- Create tutorial from your work
- Present to next year's class (optional)

### Create Tutorials
Write step-by-step guides:
- Implementing paper from scratch
- Using specific tools/libraries
- Debugging common issues
- Optimization techniques

### Build Tools
Develop utilities for:
- Dataset preparation
- Visualization
- Evaluation
- Debugging

## License

By contributing, you agree that your contributions will be licensed under the same license as the course materials (MIT License).

## Questions?

Contact:
- **Instructor:** [Email]
- **Course Forum:** [Link]
- **GitHub Issues:** [Link]

## Acknowledgments

Thank you to all contributors who help make this course better!

### Hall of Fame (Top Contributors)

*Coming soon - be the first!*

---

**Ready to contribute? Great! Here's your first step:**

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/computer-vision-course.git`
3. Find something to improve (check Issues for ideas)
4. Make your contribution
5. Submit a Pull Request

**Every contribution, no matter how small, makes a difference! 🌟**
