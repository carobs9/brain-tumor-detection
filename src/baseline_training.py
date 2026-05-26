import pandas as pd
import numpy as np
from src.utils import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import cv2

# LOADING 

training_dir = r"C:\Users\rqg886\Desktop\brain-tumor-detection\data\raw\Training"
testing_dir_no_tumor = r"C:\Users\rqg886\Desktop\brain-tumor-detection\data\raw\Testing\no_tumor"
testing_dir_tumor = r"C:\Users\rqg886\Desktop\brain-tumor-detection\data\raw\Testing\pituitary_tumor"

classes = {
    "glioma_tumor": 0,
    "meningioma_tumor": 1,
    "no_tumor": 2,
    "pituitary_tumor": 3,
}
inverted_classes = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}


X, Y = load_data(training_dir, classes)

# PROCESSING 

# turn images into numpy arrays
X = np.array(X)
Y = np.array(Y)

print("Shapes: ", X.shape, Y.shape)

# this line of code reshapes each image into a 1D vector 
# (n, 200, 200) --> (n, 200*200) = (n, 40000)
X_updated = X.reshape(len(X), -1)

# SPLIT THE DATA
# Split data into train and test

xtrain, xtest, ytrain, ytest = train_test_split(X_updated, 
                                                Y,
                                                random_state=10,
                                                test_size=.2)

print("Train shape: ", xtrain.shape, "Test shape: ", xtest.shape)

# SCALING
# Use minmax scaling to bring feature values to less than or equal to 1

scale = xtrain.max()
xtrain = xtrain / scale
xtest = xtest / scale

print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())

# TRAIN BASELINE MODEL 1
# SVM

from sklearn.svm import SVC
#sv = SVC() # define model
#sv.fit(xtrain, ytrain) # fit model on training data

# TRAIN BASELINE MODEL 1
# Logistic regression

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(C=0.1, max_iter=1000) # define model
lg.fit(xtrain, ytrain) # fit model on training data


# EVALUATION
#print("Training score: ", sv.score(xtrain, ytrain))
#print("Testing score: ", sv.score(xtest, ytest))

print("Training score: ", lg.score(xtrain, ytrain))
print("Testing score: ", lg.score(xtest, ytest))

# PREDICTION
#pred = sv.predict(xtest)
pred = lg.predict(xtest)
print(classification_report(ytest, pred, target_names=[inverted_classes[i] for i in range(4)]))
print(confusion_matrix(ytest, pred))

misclass = np.where(ytest!=pred)
print(misclass)

print("Total misclassified samples: ", len(misclass[0]), len(misclass[0])/len(pred)*100, " %")

# TESTING 

test_root = r"C:\Users\rqg886\Desktop\brain-tumor-detection\data\raw\Testing"
plt.figure(figsize=(16, 12))
c = 1
for cls in classes:                                # 4 classes
    folder = os.path.join(test_root, cls)
    for i in os.listdir(folder)[:4]:               # 4 samples each → 4x4 grid
        plt.subplot(4, 4, c)
        img = cv2.imread(os.path.join(folder, i), 0)
        img1 = cv2.resize(img, (200, 200)).reshape(1, -1) / scale
        pred = lg.predict(img1)
        plt.title(inverted_classes[pred[0]])
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        c += 1
plt.tight_layout()
plt.show()
