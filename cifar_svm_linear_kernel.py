import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, svm
from joblib import dump, load

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifar10_plot(im):
    im_r = im[0:1024].reshape(32, 32)
    im_g = im[1024:2048].reshape(32, 32)
    im_b = im[2048:].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))

    plt.imshow(img) 
    plt.show()


classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


x = unpickle('data_batch_1')
data = np.array(x[b'data'])
labels = x[b'labels']

# data = []
# labels = []

for i in range(2,6):
    x = unpickle('data_batch_{}'.format(i))
    new_data = np.array(x[b'data'])
    data = np.concatenate((data, new_data))
    labels = np.concatenate((labels, x[b'labels']))

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, labels, test_size=0.2)

# clf = neighbors.KNeighborsClassifier()
clf = svm.SVC(kernel='linear')
clf.fit(xtrain, ytrain)
accu = clf.score(xtest, ytest)
print("Accuracy {}".format(accu))

dump(clf, 'linear.joblib') 

# clf = load('linear.joblib') 

example_image = xtest[10, :].reshape(1,-1)

prediction = clf.predict(example_image)
print(prediction[0] == ytest[10])
print("Sample Image {}".format(classesName[ytest[10]]))
print("Predicted Image {}".format(classesName[prediction[0]]))

img = xtest[10, :]
cifar10_plot(img)