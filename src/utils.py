import os
import cv2


def load_data(training_dir, classes):
    
    X = [] # list of images
    Y = [] # classes associated to the training images 

    training_dir = training_dir
    classes = classes

    for cls in classes: # loops through the defined classes (no_tumor / pituitary_tumor)
        pth = os.path.join(training_dir, cls)
        print(pth)
        for j in os.listdir(pth):
            img = cv2.imread(os.path.join(pth, j), 0) # read image 
            img = cv2.resize(img, (200, 200)) # resize
            X.append(img) # append image
            Y.append(classes[cls]) # append class

    return X,Y