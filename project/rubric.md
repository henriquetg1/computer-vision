# Final Project Grading Rubric

**Total Points:** 50% of final grade (distributed as shown below)

---

## Proposal (5% of final grade)

| Criterion | Excellent (4-5) | Good (3) | Satisfactory (2) | Needs Work (0-1) |
|-----------|----------------|----------|------------------|------------------|
| **Problem Definition** | Clear, well-motivated problem with specific scope | Problem identified but could be more specific | Vague problem statement | Unclear or missing |
| **Literature Review** | Comprehensive survey of related work | Covers main related work | Minimal coverage | Missing or inadequate |
| **Proposed Approach** | Detailed technical plan with clear integration of CV+DL | Basic approach outlined | Vague technical details | Approach unclear |
| **Dataset Plan** | Specific dataset identified with analysis | Dataset identified | Generic dataset mention | No dataset plan |
| **Timeline** | Realistic, detailed timeline with milestones | Basic timeline | Overly optimistic or vague | Missing timeline |

---

## Midterm Checkpoint (5% of final grade)

| Criterion | Excellent (4-5) | Good (3) | Satisfactory (2) | Needs Work (0-1) |
|-----------|----------------|----------|------------------|------------------|
| **Progress** | Significant progress, on track | Good progress, minor delays | Some progress, behind schedule | Minimal progress |
| **Demo** | Working demo with clear results | Basic functionality demonstrated | Partial demo | No demo |
| **Presentation** | Clear, well-organized, good visuals | Decent presentation | Basic presentation | Poor presentation |
| **Problem Solving** | Shows good problem-solving approach | Addresses some challenges | Acknowledges challenges | Ignores problems |
| **Revised Plan** | Realistic updated plan based on progress | Some adjustments | Minimal replanning | No adaptation |

---

## Final Code Repository (25% of final grade)

### A. Code Quality (10 points)

| Criterion | Excellent (9-10) | Good (7-8) | Satisfactory (5-6) | Needs Work (0-4) |
|-----------|------------------|------------|-------------------|------------------|
| **Organization** | Clean structure, modular design | Organized but could be better | Basic organization | Messy, hard to navigate |
| **Documentation** | Excellent comments, docstrings, README | Good documentation | Basic documentation | Poor or missing docs |
| **Naming** | Clear, consistent variable/function names | Mostly clear names | Some unclear names | Confusing names |
| **Best Practices** | Follows Python best practices, PEP 8 | Mostly follows standards | Some violations | Many violations |
| **Reproducibility** | Easy to reproduce, seeds set, deps clear | Mostly reproducible | Partially reproducible | Cannot reproduce |

**Code Quality Checklist:**
- [ ] Clear directory structure
- [ ] requirements.txt or environment.yml
- [ ] README with setup instructions
- [ ] Docstrings for functions/classes
- [ ] Inline comments for complex logic
- [ ] Configuration files (not hardcoded)
- [ ] Consistent naming convention
- [ ] No dead code or commented-out blocks
- [ ] Error handling where appropriate
- [ ] Version control usage (meaningful commits)

### B. Classical CV Implementation (5 points)

| Criterion | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (0-2) |
|-----------|--------------|----------|------------------|------------------|
| **Technique Selection** | Appropriate techniques for problem | Reasonable choices | Questionable choices | Poor choices |
| **Implementation** | Correct, efficient implementation | Mostly correct | Basic implementation | Incorrect or buggy |
| **Integration** | Well-integrated with DL components | Some integration | Loosely connected | Not integrated |

**Expected Classical CV Components (select relevant):**
- Image preprocessing (enhancement, normalization)
- Edge/corner detection
- Feature extraction (SIFT, HOG, etc.)
- Image filtering/smoothing
- Color space analysis
- Geometric transformations
- Morphological operations

### C. Deep Learning Implementation (5 points)

| Criterion | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (0-2) |
|-----------|--------------|----------|------------------|------------------|
| **Architecture** | Well-designed, justified architecture | Reasonable architecture | Basic architecture | Poor architecture |
| **Training** | Proper training procedure, monitoring | Good training setup | Basic training | Poor training |
| **Optimization** | Well-tuned hyperparameters | Decent tuning | Minimal tuning | No tuning |

**Expected DL Components:**
- Appropriate network architecture
- Proper data loading and augmentation
- Training/validation split
- Loss function selection
- Optimizer choice and tuning
- Learning rate scheduling
- Regularization techniques
- Gradient checking/monitoring
- Model checkpointing

### D. System Functionality (5 points)

| Criterion | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (0-2) |
|-----------|--------------|----------|------------------|------------------|
| **Completeness** | Full pipeline works end-to-end | Most features work | Basic functionality | Many bugs/missing features |
| **Robustness** | Handles edge cases well | Mostly robust | Some error handling | Crashes easily |
| **Performance** | Efficient implementation | Reasonable performance | Slow but works | Very slow or doesn't work |

---

## Final Presentation (10% of final grade)

| Criterion | Excellent (9-10) | Good (7-8) | Satisfactory (5-6) | Needs Work (0-4) |
|-----------|------------------|------------|-------------------|------------------|
| **Clarity** | Crystal clear, easy to follow | Clear presentation | Somewhat clear | Confusing |
| **Content** | All key points covered, appropriate depth | Good coverage | Missing some details | Incomplete |
| **Visuals** | Excellent slides, figures, demos | Good visuals | Basic visuals | Poor visuals |
| **Demo** | Impressive live demo or video | Working demo | Partial demo | Demo fails |
| **Time Management** | Perfect timing (10 min) | Within time (±1 min) | Over/under by 2-3 min | Way off |
| **Q&A** | Handles questions expertly | Answers most questions | Some difficulty | Cannot answer |
| **Team Coordination** | Seamless transitions, balanced | Good teamwork | Uneven participation | Poor coordination |

**Presentation Structure:**
1. **Introduction (2 min):** Problem, motivation, contribution
2. **Approach (3 min):** Technical methods, architecture
3. **Results (3 min):** Quantitative + qualitative results
4. **Demo (1 min):** Live or video demonstration
5. **Conclusion (1 min):** Learnings, future work

---

## Final Report (10% of final grade)

### A. Writing Quality (3 points)

| Criterion | Excellent (3) | Good (2) | Needs Work (0-1) |
|-----------|--------------|----------|------------------|
| **Clarity** | Clear, concise, professional | Mostly clear | Unclear or verbose |
| **Grammar** | No errors | Few minor errors | Many errors |
| **Structure** | Well-organized, logical flow | Good structure | Poor organization |

### B. Technical Content (7 points)

| Section | Points | Criteria |
|---------|--------|----------|
| **Abstract** | 0.5 | Concise summary of problem, approach, results |
| **Introduction** | 1 | Clear motivation, problem statement, contributions |
| **Related Work** | 1 | Comprehensive survey, comparison with your approach |
| **Methodology** | 2 | Detailed technical description, clear diagrams |
| **Experiments** | 1.5 | Proper experimental setup, clear description |
| **Results** | 1.5 | Quantitative+qualitative results, visualizations |
| **Discussion** | 1 | Insightful analysis, limitations, future work |
| **Conclusion** | 0.5 | Summary of contributions and learnings |
| **References** | 0.5 | Proper citations, sufficient references |

**Report Checklist:**
- [ ] 6-8 pages (excluding references)
- [ ] Conference paper format (IEEE/NeurIPS)
- [ ] All required sections included
- [ ] High-quality figures and tables
- [ ] Proper citations (no plagiarism)
- [ ] Proofreading completed
- [ ] PDF format

---

## Detailed Evaluation Criteria

### Technical Depth (Overall Assessment)

**Exceptional (A+ range):**
- Novel combination of techniques
- Sophisticated implementation
- Insightful analysis
- Publication-quality work

**Strong (A range):**
- Appropriate techniques well-implemented
- Good integration of CV and DL
- Thorough evaluation
- Professional presentation

**Satisfactory (B range):**
- Basic techniques correctly applied
- Functional implementation
- Adequate evaluation
- Complete deliverables

**Needs Improvement (C range):**
- Limited techniques
- Implementation issues
- Weak evaluation
- Incomplete deliverables

**Insufficient (D/F range):**
- Minimal effort
- Major technical errors
- Missing key components
- Poor quality overall

### Results Quality

**Excellent:**
- Beats baselines significantly
- Comprehensive evaluation with multiple metrics
- Insightful failure analysis
- Statistical significance testing

**Good:**
- Competitive with baselines
- Multiple metrics reported
- Some failure analysis
- Reasonable comparison

**Satisfactory:**
- Basic results shown
- At least one metric
- Minimal analysis
- Simple comparison

**Poor:**
- Unclear results
- No baselines
- No analysis
- Incomplete evaluation

---

## Bonus Points (up to +5%)

Can push grade from 50% to 55% (10% improvement on project component):

- **Novel contribution (2%):** Publishable insight or new technique
- **Code quality (1%):** Comprehensive unit tests, CI/CD setup
- **Deployment (1%):** Working web demo or mobile app
- **Open source (1%):** Well-documented, reusable code repository
- **Dataset contribution (1%):** Create and release new dataset
- **Educational value (1%):** Tutorial or blog post explaining your work

---

## Common Issues and Deductions

| Issue | Deduction |
|-------|-----------|
| Late submission | -10% per day (max 3 days) |
| Code doesn't run | -20% |
| No demo | -10% |
| Plagiarism (code/report) | 0% + academic integrity violation |
| Missing team member contribution | Individual grades adjusted |
| Incomplete documentation | -10% |
| No GitHub repository | -15% |
| Results not reproducible | -15% |

---

## Self-Assessment (Optional)

Before submitting, evaluate your project:

| Component | Self-Rating (1-5) | Evidence |
|-----------|-------------------|----------|
| Code quality | | |
| Technical depth | | |
| Results | | |
| Presentation | | |
| Report | | |
| Overall effort | | |

**What went well:**

**What could be improved:**

**Key learnings:**

---

## Peer Evaluation (Confidential)

Each team member rates others:

**Team Member:** _______________

| Criterion | Rating (1-5) | Comments |
|-----------|-------------|----------|
| Contribution to code | | |
| Participation in meetings | | |
| Communication | | |
| Problem solving | | |
| Overall teamwork | | |

Extreme discrepancies may result in individual grade adjustments.

---

## Final Grade Calculation

**Project Component (50% of course grade):**

- Proposal: ____/5%
- Midterm Checkpoint: ____/5%
- Code Repository: ____/25%
- Presentation: ____/10%
- Report: ____/10%
- Bonus: ____/+5% (optional)

**Project Total: ____/50%**

---

## Feedback Template (For Instructor)

**Team:** _______________  
**Project:** _______________

**Strengths:**
1.
2.
3.

**Areas for Improvement:**
1.
2.
3.

**Overall Comments:**

**Recommended Future Steps:**

**Grade: ____/50%**

---

*This rubric ensures fair, comprehensive evaluation of all project aspects. Questions about grading should be directed to the instructor within one week of grade release.*
