#Understanding the output of learning_curve
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#example-model-selection-plot-learning-curve-py
#https://discussions.udacity.com/t/understanding-the-output-of-learning-curve/180204
from sklearn.linear_model import LinearRegression
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.cross_validation import KFold
import numpy as np

size = 1000
cv = KFold(size,shuffle=True)
print "cv", cv
score = make_scorer(explained_variance_score)

X = np.reshape(np.random.normal(scale=2,size=size),(-1,1))
y = np.array([[1 - 2*x[0] +x[0]**2] for x in X])

def plot_curve():
    reg = LinearRegression()
    reg.fit(X,y)
    print reg.score(X,y)

    # TODO: Create the learning curve with the cv and score parameters defined above.
    train_sizes, train_scores, test_scores = learning_curve(reg, X,y,cv=cv, train_sizes = np.linspace(0.1, 1.0, 10))
    # TODO: Plot the training and testing curves.
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.plot(train_sizes, train_scores, 'o-', color="b",
             label="Training score")
    plt.plot(train_sizes, test_scores, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    # Sizes the window for readability and displays the plot.
    plt.ylim(-.1,1.1)
    plt.show()
