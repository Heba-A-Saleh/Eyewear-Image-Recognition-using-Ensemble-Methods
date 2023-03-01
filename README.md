# Eyewear-Image-Recognition-using-Ensemble-Methods (Bagging, Boosting and Stacking Algorithms)

This repository contains code to perform basic image recognition tasks using various machine learning algorithms, including decision trees, bagging, boosting, and stacking. The focus is on eyewear prediction.

## Dataset

The dataset includes X_train, y_train, X_test, and y_test, which are provided in the template notebook.

## Decision Tree

We start by creating a decision tree classifier with parameters of random_state=0 and max_depth=2. We report the training and test accuracy of the model.

## Bagging Models

We then move on to bagging, which is a way to get a diverse set of models by using the same algorithm but training them on different training sets. We use a decision tree classifier with parameters of random_state=0 and max_depth=2 for bagging. We create two different bagging classifiers with different sampling techniques:

### Bagging Classifier 1

This bagging classifier keeps all training instances but samples features without replacement (maximum features = 40). We use random_state=0 for the bagging model as well. We then find the optimal number of models by increasing the number of models from 1 to 40 and calculating the test accuracy of each bagging model. We plot the test accuracy of bagging models and report the best number of models and the training and test accuracy of the bagging model for the best number of models.

### Bagging Classifier 2

This bagging classifier samples training instances with replacement (maximum samples=1.0) and samples features without replacement (maximum features = 40). We use random_state=0 for the bagging model as well. We then calculate the test accuracy of the bagging model as we increase the number of models from 1 to 40. We plot the test accuracy of bagging models and report the best number of models and the training and test accuracy of the bagging model for the best number of models.

We compare the decision tree classifier and the best bagging classifiers in terms of accuracy, bias, and variance. We also compare the two bagging models we found in terms of accuracy, bias, and variance and discuss how sampling features and sampling instances affected the performance of the models.

## Boosting

We then move on to boosting, which is an ensemble method that can combine several weak learners into a strong learner. We create an AdaBoost classifier with decision trees with parameters of random_state=0 and max_depth=2. We use random_state=0 for the boosting model. We then find the optimal number of models by increasing the number of models from 1 to 50 and calculating the train and test accuracy of each boosting model. We plot the train and test accuracy of boosting models and report the best number of models and the train and test accuracy of the boosting model for the best number of models.

## Stacking

Finally, we use stacking to combine multiple models. We split X_train and y_train into two sets using train_test_split function with parameters of random_state=0, test_size=0.5. We name resulting data sets as train_set_1, train_set_2, train_y_1, and train_y_2 respectively. 
