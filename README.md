# Machine Learning Specialization – Course 1 Labs

Practice labs from **Supervised Machine Learning: Regression and Classification** (Course 1 of the Machine Learning Specialization by Andrew Ng, DeepLearning.AI & Stanford on Coursera).

In these labs I built two core ML algorithms **from scratch using only NumPy** (no scikit-learn), to understand how they really work.

## Projects

| Week | Notebook | Algorithm | Problem |
|------|----------|-----------|---------|
| 2 | `C1_W2_Linear_Regression.ipynb` | Linear Regression (one variable) | Predict restaurant profit from city population |
| 3 | `C1_W3_Logistic_Regression.ipynb` | Logistic Regression + Regularization | Predict university admission and microchip quality |

---

## Week 2: Linear Regression

**Goal:** Help a restaurant franchise choose the best city for a new branch, using data from 97 cities (population vs. monthly profit).

**What I implemented:**
- `compute_cost`: the squared-error cost function
- `compute_gradient`: the gradient for `w` and `b`
- Batch gradient descent to learn the best parameters

**Results:**
- Cost with the starting values (`w=2, b=1`): **75.203**
- After 1500 iterations (learning rate `0.01`), the cost dropped to about **4.49**
- Learned parameters: `w ≈ 1.166`, `b ≈ -3.630`

---

## Week 3: Logistic Regression

### Part 1: Student admission (classification)
Predict if a student is admitted using two exam scores.

**What I implemented:**
- `sigmoid`
- `compute_cost` (logistic loss)
- `compute_gradient`
- `predict` (threshold at 0.5)

**Results:** after 10,000 iterations (learning rate `0.001`), the cost went from 0.96 to about **0.30**, with a training accuracy of **92%**.

### Part 2: Microchip quality (regularized logistic regression)
Predict if a microchip passes quality assurance, based on two test results. The data cannot be separated by a straight line, so I:
- used **feature mapping** to create polynomial features up to the 6th power (2 features became 27)
- added **regularization** to reduce overfitting
- implemented `compute_cost_reg` and `compute_gradient_reg`

**Results:** with `lambda = 0.01`, the model reached a training accuracy of about **82%** and learned a non-linear decision boundary.

---

## Key concepts I learned
- Cost functions and gradient descent
- Sigmoid function and logistic loss
- Decision boundaries
- Overfitting and regularization
- Vectorized thinking with NumPy

## Tools
Python · NumPy · Matplotlib · Jupyter Notebook

## How to run
1. Clone this repository.
2. Install the packages: `pip install numpy matplotlib jupyter`
3. Open a notebook with `jupyter notebook`.
4. Run the cells in order.

## Credits
Labs and datasets are from the [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) by DeepLearning.AI and Stanford Online. The code between `START CODE HERE` and `END CODE HERE` is my own implementation.

## About me
I am a university student learning data analytics and machine learning. I am working through the Machine Learning Specialization and building projects as I learn.
