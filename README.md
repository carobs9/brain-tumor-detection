# BRAIN TUMOR CLASSIFICATION

## The Data

The dataset used for training the models can be found in this GitHub repository: https://github.com/sartajbhuvaji/brain-tumor-classification-dataset.

The data contains training and testing MRI images with size 256x256x3.

The data is already split into training (2.870 images, 80%) and testing (394 images, 20%) with the following distribution:

| Class | Training | % | Testing | % | Total | % |
|---|---:|---:|---:|---:|---:|---:|
| glioma_tumor | 826 | 28.8% | 100 | 25.4% | 926 | 28.4% |
| meningioma_tumor | 822 | 28.6% | 115 | 29.2% | 937 | 28.7% |
| no_tumor | 395 | 13.8% | 105 | 26.6% | 500 | 15.3% |
| pituitary_tumor | 827 | 28.8% | 74 | 18.8% | 901 | 27.6% |
| **Total** | **2,870** | 100% | **394** | 100% | **3,264** | 100% |

We can see that the "no tumor" category is under represented in the training set in relation to the rest of the categories, indicating **class imbalance**. This is something important to consider when training a model, and it has been accounted for by using class weights during training.

## The Task

The main task is to classify MRI brain images into four different classes:

- Glioma tumor
- Meningioma tumor
- No tumor
- Pituitary tumor

## The Approach

To perform the classification task, different model architectures with different learning rates and augmentation techniques have been trained and compared. 

### Model Architectures

Two different pre-trained architectures have been used to solve the classification task: EfficientNetV2S and ResNet50. The reason I have used two different architectures is to compare the accuracy of the results.

### Transfer Learning

To ensure better classification results:

- The original classifier has been removed for both architectures (include_top=False). 

- This was replaced by my own head: GlobalAveragePooling → Dense(256, relu) → Dropout(0.4) → Dense(128, relu) → Dropout(0.3) → Dense(4, softmax) for the 4 tumor classes.

### Partial Fine-tuning

Most of the backbone of the models was frozen, keeping the original parameters and saving training time. The final stage was unfrozen so those laters adapt to the specific MRI training images.

- For EfficientNet, block6 and the top layer were unfrozen.

- For ResNet50, conv5 was unfrozen.

Both models ended up with approximately 15M trainable parameters.

### Class Imbalance

As mentioned earlier, the training data contains less "no tumor" examples in comparison to the rest of the classes. This is typically referred to as class imbalance, and it can lead to a biased model. To solve this, I calculated specific class weights for each class. This allows to give a higer penalty for wrong guesses on the minority class (no tumor), and a lower penalty for the majority class. In other words, this method increases the importance of correctly predicting instances from the minority class.

The given class weights were the following:

- Glioma: 0,86
- Meningioma: 0,87
- No tumor: 1,81
- Pituitary: 0,86

### Augmentation

### Learning Rate


## Interpretability: SHAP

## Improvements

The class distribution differs between splits (no_tumor is 13.8% of train but 26.6% of test) — that mismatch likely contributes to the val→test accuracy gap seen after training.


Also, implement predict_with_threshold to avoid false negatives. Read this resource: https://medium.com/data-science/achieve-better-classification-results-with-classificationthresholdtuner-39c5d454637e