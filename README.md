# NAIVE-BAYES-Iris-Dataset
## Overview
Naive-Bayes is a supervised-learning algorithms whose main assumption is independent and uncorrelated features. The Naive-Bayes(NB) classifier makes predictions on the target variables with the help of the indepedent variables. It is Naive because it uses the independent and uncorrelated assumptions of the features (For example; A living thing is called a bird if it posseses features like wings, beak, feathers, e.t.c. which intuitively depends on each other. However, the NB classifier assumes that these features independently contributes to the probability that the living-thing is classified as a bird). The Iris dataset consists of different(3) species of flower with 150 records categorized by independent features such as SepalLengthCm, PetalLengthCm, SepalWidthCm, and PetalWidthCm. The other feature is the "Species" which consist of 3-categories(virginica, versicolor and setosa).

## Motivation
Is there a simple machine learning classification algorithm that converges faster when training our model?. Can this be applied on a simple dataset to perform a classification and obtain its accuracy?.
## Goal 
Implement a Naive-Bayes Classifier algorithm on the Iris Dataset to estimate a posterior probability of a feature(Species) given some other features(SepalLengthCm, PetalLengthCm, SepalWidthCm, and PetalWidthCm).

![image](https://user-images.githubusercontent.com/54149747/109902872-a4e35a80-7c60-11eb-84f6-4dd258e79063.png)

## Data Collection/Preparation

The data for this project is obtained through Kaggle download using the following link : https://www.kaggle.com/uciml/iris. 
With the help of python jupyter notebook, libraries such as Pandas, Numpy, Matplotlib, Seaborn are used for data preparation and visualization

The First 5 rows of the Iris Data
![image](https://user-images.githubusercontent.com/54149747/109911249-51c4d400-7c6f-11eb-977e-1253d02b527a.png)

Checking for Missing Values: Using HeatMap(Which shows there was no missing values)
![image](https://user-images.githubusercontent.com/54149747/109911539-e3ccdc80-7c6f-11eb-9e44-dfb2770a4882.png)

Although, as explained earlier, the Naive Bayes assumes the independent variables are uncorrelated. We try to see whether correlation exist as shown in the image below.
![image](https://user-images.githubusercontent.com/54149747/109911980-c3515200-7c70-11eb-957f-1829285b733b.png)

Countplot for the Dependent Variable(Species)
![image](https://user-images.githubusercontent.com/54149747/109912152-25aa5280-7c71-11eb-8112-27b67825f105.png)

Barplot of Species against SepalLengthCm
![image](https://user-images.githubusercontent.com/54149747/109912364-8043ae80-7c71-11eb-938c-c7126001e998.png)

Boxplot of Species against SepalLengthCm
![image](https://user-images.githubusercontent.com/54149747/109912409-9a7d8c80-7c71-11eb-9b4d-aac0f9075f5a.png)

## Technical Aspect
The dataset is splitted into training set(40 samples per category) to train our model and test set (10 samples per category) to validate the performance of our model.

The "GaussianNB" (which is assigned to the classifier) is imported from the "sklearn.naive_bayes" library as we intended to use the Gaussian model(user choice) and then perform fitting on the training.

The "classifier.predict()" is used to predict our test data since it is assumed that our model has been trained.

## Result/Performance
![image](https://user-images.githubusercontent.com/54149747/109916221-d700b680-7c78-11eb-9901-eb4e23d84d20.png)


The image above shows the accuracy of our result (0.966667), which is 96.7%. This means that our model has a high accuracy.
The Matrix that follows is a confusion matrix(see https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62) which shows that of the test data(10 samples of each three flower category), our model predicted 29 correctly and only 1(FALSE NEGATIVE) was incorrectly predicted.

## Conclusion
In this work, the Naive-Bayes Classifier algorithm has been successfully implemented to predict Iris-flower categories with a very good accuracy using the iris-dataset with three different categories of flower.




