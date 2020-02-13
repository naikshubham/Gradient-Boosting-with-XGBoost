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

### Decision Tree
- Decision Tree has a single question being asked at each decision node, and only 2 possible choices, at the very bottom of each decision tree, there is a single possible decision.

#### Decision trees as base learners
- Any individual learning algorithm in an ensemble algorithm is a **base learner**. XGBoost itself is an ensemble learning method, it uses the outputs of many models for a final prediction.
- Decision tree is a learning method that involves a tree-like graph to model either continuos or categorical choice given some data.
- It is composed of a series of binary decisions (yes/no or true/false questions) that when answered in succession ultimately yield a prediction about the data at hand(these predictions happen at the leaves of the tree).
- Decision trees are constructed iteratively (i.e, one binary decision at a time) until some stopping criterion is met(the depth of the tree reaches some pre-defined maximum value, for example)
- During construction, the tree is built one split at a time, and the way that a split is selected (i.e, what feature to split on and where in the feature's range of values to split) can vary, but involves choosing a split point that segregates the target values better (puts each target category into buckets that are increasingly dominated by just one category) until all (or nearly all) values within a given split are exclusively of one category or another.
- Using this process, each leaf of the decision tree will have a single category in the majority, or should be exclusivley of one category.

#### Individual decision trees tend to overfit
- Individual decision trees in general are low bias, high variance learning models.i.e, they are very good at learning realtionships within any data you train them on, but they tend to overfit the data we use to train them on and usually generalize to new data poorly.

#### CART : Classification and Regression Trees
- **XGBoost** uses a slighlty different decision tree called **classification and regression tree, or CART**. Whereas for the decision trees described above the leaf nodes always contain decision values, **`CART trees contain a real-valued score in each leaf`**, regardless of whether they are used for classification or regression.
- The real-valued scores can then be thresholded to convert into categories for classification problems if necessary.
















































