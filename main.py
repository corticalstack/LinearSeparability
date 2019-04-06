import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.datasets import load_wine
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap


class LinearSeparability:
    def __init__(self):
        self.wine = None
        self.df_wine = None
        self.X = None
        self.y = None
        self.random_state = 20
        self.class_colours = np.array(["red", "green", "blue"])
        self.target = None
        self.load_data()
        self.set_data()
        self.scatter_matrix()
        self.scatter_target()
        self.convex_hull()
        self.perceptron()
        self.svm()
        self.rbf()

    @staticmethod
    def confusion_matrix(y, y_pred):
        cm = confusion_matrix(y, y_pred)
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        class_names = ['Negative', 'Positive']
        plt.title('Perceptron Confusion Matrix - Entire Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        s = [['TN', 'FP'], ['FN', 'TP']]

        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
        plt.show()

    def show_boundary(self, x, y, clf):
        plt.clf()

        if isinstance(x, pd.DataFrame):
            x_set, y_set = x.values, y.values
        else:
            x_set, y_set = x, y

        x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
        plt.contourf(x1, x2, clf.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                     alpha=0.75, cmap=ListedColormap(('navajowhite', 'darkkhaki')))
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                        c=self.class_colours[i], label=j)
        plt.title('Perceptron Classifier - Decision boundary')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.legend()
        plt.show()

    def load_data(self):
        self.wine = load_wine()

    def set_data(self):
        self.df_wine = pd.DataFrame(self.wine.data, columns=self.wine.feature_names)
        self.X = self.df_wine.values
        self.y = pd.Series(self.wine.target)

    def scatter_matrix(self):
        scatter_matrix(self.df_wine.iloc[:, 0:4], figsize=(15, 11))
        plt.show()

    def scatter_target(self):
        plt.clf()
        plt.figure(figsize=(10, 6))
        names = self.wine.target_names
        plt.title(self.df_wine.columns[0] + ' vs ' + self.df_wine.columns[1])
        plt.xlabel(self.df_wine.columns[0])
        plt.ylabel(self.df_wine.columns[1])
        for i in range(len(names)):
            bucket = self.df_wine[self.y == i]
            bucket = bucket.iloc[:, [0, 1]].values
            plt.scatter(bucket[:, 0], bucket[:, 1], label=names[i], c=self.class_colours[i])
        plt.legend()
        plt.show()

    def convex_hull(self):
        plt.clf()
        plt.figure(figsize=(10, 6))
        names = self.wine.target_names
        plt.title(self.df_wine.columns[0] + ' vs ' + self.df_wine.columns[1])
        plt.xlabel(self.df_wine.columns[0])
        plt.ylabel(self.df_wine.columns[1])
        for i in range(len(names)):
            bucket = self.df_wine[self.y == i]
            bucket = bucket.iloc[:, [0, 1]].values
            hull = ConvexHull(bucket)
            plt.scatter(bucket[:, 0], bucket[:, 1], label=names[i], c=self.class_colours[i])
            for j in hull.simplices:
                plt.plot(bucket[j, 0], bucket[j, 1], c=self.class_colours[i])
        plt.legend()
        plt.show()

    def perceptron(self):
        perceptron = Perceptron(max_iter=100, tol=1e-3, random_state=self.random_state)
        _x = self.df_wine.iloc[:, [0, 1]]
        # Boolean cast classes other than 1 to 0
        _y = (self.y == 1).astype(np.int)
        perceptron.fit(_x, _y)
        predicted = perceptron.predict(_x)
        self.confusion_matrix(_y, predicted)
        self.show_boundary(_x, _y, perceptron)

    def svm(self):
        svm = SVC(C=1.0, kernel='linear', random_state=self.random_state)
        _x = self.df_wine.iloc[:, [0, 1]]
        # Boolean cast classes other than 1 to 0
        _y = (self.y == 1).astype(np.int)
        svm.fit(_x, _y)
        predicted = svm.predict(_x)
        self.confusion_matrix(_y, predicted)
        self.show_boundary(_x, _y, svm)

    def rbf(self):
        sc = StandardScaler()
        _x = self.df_wine.iloc[:, [0, 1]]
        # Boolean cast classes other than 1 to 0
        _y = (self.y == 1).astype(np.int)
        _x = sc.fit_transform(_x)
        svm = SVC(kernel='rbf', random_state=self.random_state)
        svm.fit(_x, _y)
        predicted = svm.predict(_x)
        self.confusion_matrix(_y, predicted)
        self.show_boundary(_x, _y, svm)


linearSeparability = LinearSeparability()
