# 🧠 Linear Separability Visualization

A Python tool for visualizing linear separability concepts using machine learning classifiers on the Wine dataset.

## 📝 Description

This repository demonstrates the concept of linear separability in machine learning by applying different classification algorithms to the Wine dataset. It visualizes data distributions, decision boundaries, and classification performance metrics to help understand how different algorithms separate data points in feature space.

## ✨ Features

- 📊 Data visualization using scatter plots and scatter matrices
- 🔷 Convex hull visualization to show class boundaries
- 🔍 Implementation of multiple classifiers:
  - Perceptron (linear classifier)
  - Support Vector Machine with linear kernel
  - Support Vector Machine with RBF kernel (non-linear)
- 📈 Decision boundary visualization for each classifier
- 📉 Confusion matrix generation to evaluate classifier performance

## 🛠️ Prerequisites

- Python 3.6+
- Required libraries:
  ```
  pandas
  numpy
  scikit-learn
  matplotlib
  scipy
  ```

## 🚀 Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/corticalstack/LinearSeparability.git
   cd LinearSeparability
   ```

2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib scipy
   ```

3. Run the main script:
   ```bash
   python main.py
   ```

4. Observe the generated visualizations:
   - Scatter matrix of the first 4 features
   - Scatter plot of alcohol vs malic acid features with class coloring
   - Convex hull visualization of class boundaries
   - Decision boundaries and confusion matrices for each classifier

## 🧩 How It Works

The `LinearSeparability` class performs the following steps:

1. Loads the Wine dataset from scikit-learn
2. Creates visualizations of the dataset features
3. Implements three different classifiers:
   - Perceptron: A simple linear classifier
   - SVM with linear kernel: A more robust linear classifier
   - SVM with RBF kernel: A non-linear classifier
4. For each classifier, it:
   - Trains on the first two features (alcohol and malic acid)
   - Visualizes the decision boundary
   - Displays a confusion matrix to evaluate performance

## 📚 Technical Details

- The project focuses on binary classification by converting the Wine dataset (which has 3 classes) into a binary problem (class 1 vs. others)
- The visualization shows how different algorithms create decision boundaries between classes
- Demonstrates the difference between linear classifiers (Perceptron, linear SVM) and non-linear classifiers (RBF SVM)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Wine Dataset Description](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset)
- [Understanding Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [Perceptron Algorithm](https://scikit-learn.org/stable/modules/linear_model.html#perceptron)
