import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

#=========read data and define classifier==========
infile = np.load('FaceData.npz')
X = np.array(infile['Faces'])
FaceImages = np.array(infile['FaceImages'])
y = np.array(infile['y'])

kNN = KNeighborsClassifier()

#=============print some faces=================
plt.figure(figsize=(15,7))
for iImg in range(3):
    for jImg in range(8):
        plt.subplot(3,8,iImg*8+jImg+1)
        plt.imshow(FaceImages[iImg*8+jImg], cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

print(str(len(X))+' faces in total.')
print()

#==========set and calculate optimal parameters===========
'''
param = {'n_neighbors':list(range(5,100,5)),
         'weights':['uniform', 'distance']}
'''
param = {'n_neighbors':list(range(1,11,1)),
         'weights':['uniform', 'distance']}

grid_kNN = GridSearchCV(kNN, param, cv=10)
grid_kNN.fit(X,y)

print()
print('========== ignore warnings ¯\_(ツ)_/¯ ==========')
print('(...)')
print()
print('========== KNN classification ===========')
print('Best Parameters for KNN : ', end='')
print(grid_kNN.best_params_)
print('Overall Accuracy of best KNN : ', end='')
print(grid_kNN.best_score_)

#============cross validation============
from sklearn.model_selection import cross_val_score

kNN = KNeighborsClassifier(n_neighbors = 1, weights = 'uniform')
scores = cross_val_score(kNN, X, y, cv=10)
print()
print('========== cross validation for cv=10(KNN) ==========')
print('Respective Accuracy per fold(KNN) : ')
print(scores)
print('Mean Accuracy of 10 folds(KNN) : ', end='')
print(scores.mean())

#=========SVM score===========
from sklearn.svm import SVC

print()
print('========== SVM classification ===========')
sv = SVC(kernel='linear', C=1.0)
sv.fit(X,y)
scores_sv = cross_val_score(sv, X,y,cv=10)
print('Respective Accuracy per fold(SVM) : ')
print(scores_sv)
print('Mean Accuracy of 10 folds(SVM) : ', end='')
print(scores_sv.mean())

