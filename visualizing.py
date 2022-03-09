import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions

iris = load_iris()
X, y = iris.data[:, 2:], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=123,
                                                    shuffle=True)

#plotting all featues
names = ['sepal length [cm]', 'sepal width [cm]',
         'petal length [cm]', 'petal width [cm]']

fig, axes = scatterplotmatrix(iris.data[y==0], figsize=(10, 8), alpha=0.5)
fig, axes = scatterplotmatrix(iris.data[y==1], fig_axes=(fig, axes), alpha=0.5)
fig, axes = scatterplotmatrix(iris.data[y==2], fig_axes=(fig, axes), alpha=0.5, names=names)
plt.tight_layout()
#plotting the selected features
selected_names =['petal length [cm]', 'petal width [cm]']
fig2,axes2 = scatterplotmatrix(X_train[y_train==0],figsize=(10,8),alpha=0.5)
fig2,axes2 = scatterplotmatrix(X_train[y_train==1],fig_axes=(fig2,axes2),alpha=0.5)
fig2,axes2 = scatterplotmatrix(X_train[y_train==2],fig_axes=(fig2,axes2),alpha=0.5,names=selected_names)

#fitting the model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train,y_train)

#making predictions
y_pred = knn_model.predict(X_test)
value= y_test == y_pred
score = knn_model.score(X_test,y_test)
print('Test set accuarcy: {:.2f}%'.format(score*100))

#plotting Decision boundary
fig3 =  plt.figure() 
plot_decision_regions(X_train,y_train,knn_model)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()
