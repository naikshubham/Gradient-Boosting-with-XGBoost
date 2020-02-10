# Gradient-Boosting-with-XGBoost
Gradient boosting is currently one of the most popular techniques for efficient modeling of tabular datasets of all sizes. XGboost is a very fast, scalable implementation of gradient boosting, with models using XGBoost regularly winning online data science competitions and being used at scale across different industries.

### Supervised Learning
- When dealing with binary supervised learning problems, the AUC or Area Under the Receiver Operating Characteristic Curve, is the most versatile and common evaluation metric used to judge the quality of a binary classification model.It is simply the probability that a randomly chosen positive data point will have a higher rank than a randomly chosen negative data point.
- Higher AUC means a more sensitive, better performing model.
- When dealing with multi class classification problem its common to use the accuracy score and to look at the overall confusion matrix to evaluate the quality of a model.
- All supervised learning problems, including classification problems, requires that the data is structured as a table of feature vectors, where the features themselves(also called attributes or predictors) are either numeric or categorical.
- Furthermore, it is usually the case that numeric features are scaled to aid in either feature interpretation or to ensure that the model can be trained properly (for example numerical feature scaling is essential to ensure properly trained support vector machine models)
- Categorical featres should be encoded before applying supervised learning algorithms (one-hot encoding)

### XGBoost Introduction
- XGBoost is an popular machine learning library. Originally written in C++, has APIs in several languages like Python, R, Scala, Julia, Java.

#### What makes XGBoost so popular ? - It's speed and performance
- Because the core XGBoost algorithm is parallelizable, it can harness all of the processing power of modern multi-core computers. Furthermore, it is parallelizable across GPU's and across network of computers, making it feasible to train models on very large datasets on the order of hundered of millions of training examples.

#### Using XGBoost : example

```python
import xgboost as xgb
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

class_data = pd.read_csv('classification_data.csv')

X, y = class_data.iloc[:,:-1], class_data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

xg_cl.fit(X_train, y_train)

preds = xg_cl.predict(X_test)
```

































