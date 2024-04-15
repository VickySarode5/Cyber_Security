# <p align="center"> Unmasking Cyber Fraud: Enhancing Detection Through Machine Learning </p>
# 

</p>

## Overview

This project is centered on developing a cybersecurity transaction fraud detection system. It employs a range of supervised and unsupervised learning methods to examine transactional data and pinpoint potentially fraudulent activities. The objective is to detect suspicious activities in transactional data by utilizing machine learning techniques and anomaly detection algorithms. Through the analysis of patterns, outliers, and other markers, the system aids in thwarting fraudulent transactions, thereby boosting cybersecurity efforts.

**Tools:-** Excel,Python

[Datasets Used](https://docs.google.com/spreadsheets/d/1Yp_rcOS2TbVn-wHUIsCeCzkeDP7MIPLP/edit?usp=sharing&ouid=102868121048017441192&rtpof=true&sd=true )

[Python Script (Code)](cyber_security.ipynb)


### Features 

* **Data preprocessing:** Cleansing and transforming raw transactional data into a suitable format for analysis.
* **Feature engineering:** Extracting relevant features from the data to improve the performance of the fraud detection model.
* **Machine learning model development:** Building predictive models using algorithms such as logistic regression, random forest, or neural networks.
* **Anomaly detection:** Employing anomaly detection techniques to identify suspicious transactions that deviate from normal behavior.
* **Real-time monitoring:** Implementing a system for continuous monitoring of transactions to detect and flag potential fraud in real-time.


## Requirements

- Python 3

- Libraries: NumPy, pandas, Sklearn, etc.

- Jupyter Lab

## Balancing an unbalanced dataset:
```py
#So, we can do Undersampling technique to balance the datasets otherwise As you can see, this model is only predicting 0, which means it’s completely ignoring the minority class in favor of the majority class.
df_majority = sample[sample.Attack == 0]
df_minority = sample[sample.Attack == 1]
df_majority_undersample = df_majority.sample(replace = False, n = 144503, random_state = 123)#random_state it's won't shuffle if we run this multiple time
b_sample = pd.concat([df_majority_undersample, df_minority])
print(b_sample.Attack.value_counts())
b_sample.shape
```
```py
fig = plt.figure(figsize = (8,5))
b_sample.Attack.value_counts().plot(kind='bar', color= ['red','green'], alpha = 0.9, rot=0)
plt.title('Distribution of data based on the Binary attacks of our balanced dataset')
plt.show()
```
###### Result: 

![img1](https://github.com/Aathimuthu25/CYBER-SECURITY/assets/158067286/a0d9a286-bdcf-4562-83d4-7e993037fc68)


## Model evaluation:
#### Decision Tree Classifier Model
```py
ds=DecisionTreeClassifier(max_depth=3)
ds.fit(x_train,y_train)
train_pred=ds.predict(x_train)
test_pred=ds.predict(x_test)
print(accuracy_score(train_pred,y_train))
print(accuracy_score(test_pred,y_test))
```
```py
#creating list for train test accuracy
train_test = ['Train','test']
aucc = [dt_aucc,dt_test] 
plt.figure(figsize=(8, 4))
# Plot the bar graph
bars = plt.bar(train_test, aucc, color=['brown', 'skyblue'])
#Add Labels and title 
plt.xlabel('Decision Tree')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for train test')
#Show the plot
plt.show()
```

###### Result:

![img2](https://github.com/Aathimuthu25/CYBER-SECURITY/assets/158067286/b7ece884-3eb2-4516-b566-a71ff5d86941)


#### Building A Decision Tree Classifier plot 

```py
#8. building model using  decision trees classifier 
import matplotlib.pyplot as plt
from  sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(ds,feature_names=x.columns.tolist(),class_names=["0","1"],filled=True)
plt.show()
```

###### Result:

![img3](https://github.com/Aathimuthu25/CYBER-SECURITY/assets/158067286/4ba920e8-5110-40f6-9050-6050f7ddb43a)





### Random Forest classifier Model
```py
rfr=RandomForestClassifier(n_estimators=9,max_depth=5,random_state=42)
rfr.fit(x_train,y_train)
test_pred_rf=rfr.predict(x_test)
train_pred_rf=rfr.predict(x_train)
print(accuracy_score(train_pred_rf,y_train))
print(accuracy_score(test_pred_rf,y_test))
test_aucc=accuracy_score(test_pred_rf,y_test)
rf_aucc=accuracy_score(train_pred_rf,y_train)
```
```py
#creating list for train test accuracy
train_test = ['Train','test']
aucc = [rf_aucc,dt_test] 
plt.figure(figsize=(8, 4))
# Plot the bar graph
bars = plt.bar(train_test, aucc, color=['indigo', 'yellow'])
#Add Labels and title 
plt.xlabel('Random Forest')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for train test accuracy')
#Show the plot
plt.show()
```

###### Result:
![img4](https://github.com/Aathimuthu25/CYBER-SECURITY/assets/158067286/d2549bb3-7f73-483c-987c-6d9f831a9450)



### Logistic Regression Model

```py
model = LogisticRegression()
model.fit(x_train, y_train)
train_pred_logi=model.predict(x_train)
test_pred_logi = model.predict(x_test)
print(accuracy_score(train_pred_logi,y_train))
lr_aucc=accuracy_score(train_pred_logi,y_train)
print(accuracy_score(test_pred_logi,y_test))
print(accuracy_score(test_pred_logi,y_test))
lr_test=accuracy_score(test_pred_logi,y_test)
```
```py
#creating list for train test accuracy
train_test = ['Train','Test']
aucc = [lr_aucc,lr_test] 
plt.figure(figsize=(8, 4))
# Plot the bar graph
bars = plt.bar(train_test, aucc, color=['skyblue', 'olive'])
#Add Labels and title 
plt.xlabel('Logistic Regrassion')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Train Test')
#Show the plot
plt.show()
```

###### Result:
![img5](https://github.com/Aathimuthu25/CYBER-SECURITY/assets/158067286/9e97cae6-2402-4b7b-a348-baf4f55df40c)



### K-Nearest Neighbour Model 
 
```py
“K-Nearest Neighbour
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
knn_model = KNeighborsClassifier(n_neighbors=5)
# Fit the model to your training data
knn_model.fit(x_train_scaled, y_train)
# Predict labels for test data
test_pred_knn = knn_model.predict(x_test_scaled)
train_pred_knn = knn_model.predict(x_train_scaled)
print(accuracy_score(train_pred_knn,y_train))
knn_aucc=accuracy_score(train_pred_knn,y_train)
print(accuracy_score(test_pred_knn,y_test))
knn_test=accuracy_score(test_pred_knn,y_test)
```
```py
#creating list for train test accuracy
train_test = ['Train','Test']
aucc = [knn_aucc,knn_test] 
plt.figure(figsize=(8, 4))
# Plot the bar graph
bars = plt.bar(train_test, aucc, color=['peru', 'palegreen'])
#Add Labels and title 
plt.xlabel('KNN')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Train Test')
#Show the plot
plt.show()
```

###### Result:

![img6](https://github.com/Aathimuthu25/CYBER-SECURITY/assets/158067286/2c682d99-59e5-418e-8201-52ac1c73462d)



### Naive Bayes Model

```py
gnb = GaussianNB()
# fit the model
gnb.fit(x_train, y_train)
y_train_pred = gnb.predict(x_train)
y_train_pred = pd.Series(y_train_pred)
Model_data_train = pd.DataFrame(y_train)
Model_data_train.shape
Model_data_train['y_pred'] = y_train_pred
print('model accuracy-->{0:0.3f}'.format(accuracy_score(y_train,y_train_pred)))
naive_aucc=accuracy_score(y_train,y_train_pred)
# Data validation on x_test
test_pred_naive=gnb.predict(x_test)
print(accuracy_score(test_pred_naive,y_test))
naive_test=accuracy_score(test_pred_naive,y_test)

from sklearn.metrics import confusion_matrix

data_table = confusion_matrix(y_train, y_train_pred)

print('Confusion matrix\n\n', data_table)

print('\nTrue Positives(TP) = ', data_table[0,0])

print('\nTrue Negatives(TN) = ', data_table[1,1])

print('\nFalse Positives(FP) = ', data_table[0,1])

print('\nFalse Negatives(FN) = ', data_table[1,0])
data_table.shape
```

###### Result:
##### Confusion Matrix:

![img11](https://github.com/Aathimuthu25/CYBER-SECURITY/assets/158067286/6fffec5d-08ac-4a8e-a2f0-4004f00d1fa8)


##### Ploting Confusion Matrix Using Heat Map
```py
matrix = pd.DataFrame(data=data_table, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(matrix, annot=True, fmt='d', cmap='YlGnBu')
```
###### Result:

![img7](https://github.com/Aathimuthu25/CYBER-SECURITY/assets/158067286/05e000e0-4e2f-4392-9c3e-3d73daf9156c)


```py
#creating list for train test accuracy
train_test = ['Train','Test']
aucc = [naive_aucc,naive_test] 
plt.figure(figsize=(8, 4))
# Plot the bar graph
bars = plt.bar(train_test, aucc, color=['lightcoral', 'gold'])
#Add Labels and title 
plt.xlabel('Naive Bayes')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Train Test')
#Show the plot
plt.show()
```
###### Result:

![img8](https://github.com/Aathimuthu25/CYBER-SECURITY/assets/158067286/12ea043a-c3ea-43d9-8c8b-0b9d9417c08f)


### Ensamble Learning bagging Model

```py
from sklearn.ensemble import BaggingClassifier, BaggingRegressor,RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
pargrid_ada = {'n_estimators': [5,10,15,20,25,30,35,40]}
gscv_bagging = GridSearchCV(estimator=BaggingClassifier(), 
                        param_grid=pargrid_ada, 
                        cv=5,
                        verbose=True, n_jobs=-1, scoring='roc_auc')
gscv_results = gscv_bagging.fit(x_train, y_train)
gscv_results.best_params_
gscv_results.best_score_
ensm_aucc=metrics.roc_auc_score(y_train, pd.DataFrame(gscv_results.predict_proba(x_train))[1])
print(metrics.roc_auc_score(y_test, pd.DataFrame(gscv_results.predict_proba(x_test))[1]))
ensm_test=metrics.roc_auc_score(y_test, pd.DataFrame(gscv_results.predict_proba(x_test))[1])

#creating list for train test accuracy
train_test = ['Train','Test']
aucc = [ensm_aucc,ensm_test] 
plt.figure(figsize=(8, 4))
# Plot the bar graph
bars = plt.bar(train_test, aucc, color=['cyan', 'silver'])
#Add Labels and title 
plt.xlabel('Ensamble Learning Bagging')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Train Test')
#Show the plot
plt.show()

```
###### Result:

![img9](https://github.com/Aathimuthu25/CYBER-SECURITY/assets/158067286/97ccbbae-0eec-4c9f-9b2d-b140a3915762)


##### Compariosion of All The Accuracy of Each Model

```py
#Create a bar graph for knn, decision tree, random forest, and logistic regression
models = ['KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes','Ensemble'] 
accuracy_values = [knn_aucc, dt_aucc, rf_aucc, naive_aucc,ensm_aucc] 
plt.figure(figsize=(13, 5))
# Plot the bar graph
bars = plt.bar(models, accuracy_values, color=['blue', 'green', 'red', 'orange', 'skyblue'])
#Add accuracy values on top of each bar
plt.bar_label(bars, labels=[f"{acc:.2f}" for acc in accuracy_values])
#Add Labels and title 
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Different Models')
#Show the plot
plt.show()
```

###### Result:

![img10](https://github.com/Aathimuthu25/CYBER-SECURITY/assets/158067286/2fbdba19-032f-4604-91fa-75ca5dd7c199)


### Conclusion



- In our dataset we are having 41 variables then we find out our 'Y' variable who is having good relation with most of the variables. Then we have decided that whichever variables having more than 30% correlation with the 'Y' variable 'Attack' that will be our final data set for machine learning. After that the final dataset is having now 18 varibles including 'Y' varible.

- Fraud Detection System represents a critical tool in the fight against cybercrime, providing organizations with the means to protect their assets and customers from fraudulent activities. we have applied so many machine learning models to get the perfect result without ofverfitting, and finally we get the final model using Logistic regression model there we got 99% accuracy and there is no overfitting.



















