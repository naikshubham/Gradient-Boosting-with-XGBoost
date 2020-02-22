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

### Boosting
- Boosting is not an ML algorithm but an concept that can be applied to a set of ML models. Specifically, it is an ensemble meta-algorithm primarily used to reduce any given single learner's variance and to convert many weak learners into an arbitrarily strong learner.

#### Weak learner and strong learner
- A weak learner is any ML algorithm that is just slightly better than chance.So, a decision tree that can predict some outcome slighlty more accurately than pure randomness would be considered a weak learner.
- The principal insight that allows XGBoost to work is the fact that we can use boosting to convert a collection of weak learners into a strong learner.Where a strong learner is any alogrithm that can be tuned to acheive a good performance for some supervised learning problem.

#### How boosting is accomplished
- By iteratively learning a set of weak models on subsets of the data we have at hand, and weighting each of their predictions according to each weak learner's performance. We then combine all of the weak learner's predictions multiplied by their weights to obtain a single final weighted prediction that is much better than any of the individual predictions themselves. 

#### Boosting example
- Boosting using 2 decision trees. Given a specific example each tree gives a different prediction score depending on the data it sees. The prediction scores for each possibility are summed across trees and the prediction is simply the sum of the scores across both trees.

#### Model evaluation through cross-validation
- Example that shows how model evaluation using cross-validation works with XGBoost's learning api (which is different from the scikit-learn compatible API) as it has cross-validation capabilities baked in.
- Cross validation is a robust method for estimating the expected performance of a ML model on unseen data by generating many non-overlapping train/test splits on the training data and reporting the average test set performance across the data splits.

#### Cross-validation with XGBoost Example

```python
import xgboost as xgb
import pandas as pd

churn_data = pd.read_csv("classification_data.csv")
churn_dmatrix = xgb.DMatrix(data=churn_data.iloc[:,:-1], label=churn_data.month_5_stil_here) 
```

- here dataset is converted into an optimized data structure that the creators of XGBoost made that gives the package its performance and efficiency gain called a DMatrix
- When we use the **XGBoost cv object**, which is part of XGBoost's learning api we have to first explicitly convert our data into a **DMatrix**
- So that's what is done here before cross validation is ran

```python
params = {"objective":"binary:logistic", "max_depth":4} 
```

- param dict to pass into cv, this is necessary because the cv method has no idea what kind of XGBoost model we are using and expects us to provide that info as a dictionary of appropriate key value pairs.
- parameter dictionary is providing the objective function we would like to use and the maximum depth that every tree can grow to.

```python
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=4, num_boost_round=10, metrics="error", as_pandas=True)
print("Accuracy: %f %((1-cv_results["test-error-mean"]).iloc[-1]))
```

### When to use XGBoost
- We should consider using XGBoost for any supervised learning task that fits the following criteria:
- When we have a large number of training examples : a dataset that has few features and atleast 1000 examples. However, in general as long as the number of features in the training set is smaller than the number of examples we should be fine.
- XGBoost tends to do well, when we have a mixture of categorical and numerical features or when we just have numeric features.

### When to not use XGBoost
- The most important kind of problems where XGBoost is a suboptimal choice involve either those that have found success using other state-of-the-art algorithms or those that suffer from dataset size issues.
- Specificaly, XGBoost is not ideally suited for image recognition, computer vision, or natural language processing and understanding problems, as those kinds of problems can be much better tackled using deep learning approaches.
- In terms of dataset size problems, XGBoost is not suitable for small training sets(less than 100 training examples) or when the number of training examples is significantly smaller than the number of features being used for training

### XGBoost for Regression
- Regression problems involve predicting continuos or real values.
- Common regression metrics : Root mean squared error (RMSE) or Mean absolute error(MAE)
- RMSE is computed by taking the difference between the actual and the predicted values for what we are predicting, squaring those differences, computing their mean and taking that value's square root. This allows us to treat negative and positive differences equally, but tends to punish larger differences between predicted and actual values much more than smaller ones.
- MAE on other hand, simply sums the absolute differences between predicted and actual values across all of the samples we build our model on. Although MAE isn't affected by large differences as much as RMSE it lacks some nice mathematical properties that make it much less frequently used as an evaluation metric.

### Objective(loss) functions and base learners
- An objective or loss function quantifies how far off our prediction is from the actual result for a given data point. It maps the difference between the predictions and the target to a real number.
- When we construct any ML model, we do so in the hopes that it minimizes the loss function across all of the data points we pass in.That's our ultimate goal the smallest possible loss.

#### Common loss functions and XGBoost
- Loss functions have specific naming conventions in XGBoost. For regression models, the most common loss function used is called **reg linear**.
- For binary classification models the most common loss functions used are **reg logistic**, when we simply want the category of the target, and **binary logistic**, when we want the actual predicted probability of the positive class.
- So till now we used reg logistic loss function when building our classification models in XGBoost.

#### Base learners and their need
- XGBoost is an ensemble learning method composed of many individual models that are added together to generate a single prediction. Each of the individual models that are trained and combined are called base learners.
- The goal of XGBoost is to have base learners that is slightly better than random guessing on certain subsets of training examples, and uniformly bad at the remainder, so that when all of the predictions are combined, the uniformly bad predictions cancel out and those slighlty better than chance combine into a single very good prediction.
- Want base learners that when combined create final prediction that is non-linear. Each base learner should be good at distinguishing or predicting different parts of the dataset.
- Two kinds of base learners: tree and linear

#### Trees as base learners example : Using XGBoost's Scikit-learn compatible API

```python
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

boston_data = pd.read_csv('boston_housing.csv')
x, y = boston_data.iloc[:,:-1], boston_data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective='reg:linear', n_estimators=10, seed=123) #booster="gbtree" is default
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
print("RMSE : %f" % (rmse))
```

#### Linear base learners example : XGBoost's learning API only

```python
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

boston_data = pd.read_csv('boston_housing.csv')
x, y = boston_data.iloc[:,:-1], boston_data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
```

- Convert training and testing sets into Dmatrix object, as required by the XGBoost learning API.
- In params dict we specify the base learner we want i.e **gblinear** and the **reg linear** objective function

```python
DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test = xgb.DMatrix(data=X_test, label=y_test)
params = {"booster":"gblinear", "objective":"reg:linear"}

xgb_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=10)
preds = xg_reg.predict(DM_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
```

### Regularization in XGBoost
- The idea of penalizing models as they become more complex is called regularization.
- Loss functions in XGBoost are used to find models that are both accurate and as simple as they can possibly be. There are several parameters that can be tweaked in XGBoost to limit model complexity by altering the loss function.
- Regularization parameters in XGBoost : **Gamma** is a parameter for tree base learners that controls whether a given node on a base learner will split based on the expected reduction in the loss that would occur after performing the split, so that higher values lead to fewer splits.
- **Alpha** is another name for L1 regularization. However, this regularization term is a penalty on leaf weights rather than on feature weights, as is the case in linear or logistic regression.
- Higher alpha values lead to stronger L1 regularization, which causes many leaf weights in the base learners to go to 0.
- **Lambda** is another name for L2 regularization. L2 regularization is a much smoother penalty than L1 and causes leaf weights to smoothly decrease, instead of enforcing strong sparsity constraints on the leaf weights as in L1.

#### L1 regularization in XGBoost example

```python
import xgboost as xgb
import pandas as pd
boston_data = pd.read_csv("boston_data.csv")

X,y = boston_data.iloc[:,:-1], boston_data.iloc[:,-1]
```

- convert X matrix and y vector into a single optimized DMatrix object 

```python
boston_dmatrix = xgb.DMatrix(data=X, label=y)
params = {"objective":"reg:linear", "max_depth":4}
l1_params = [1, 10, 100]
rmse_l1 = []
for reg in l1_params:
    params['alpha'] = reg
    cv_results = xgb.cv(dtrain=boston_dmatrix, params=params, nfold=4, num_boost_round=10, metrics="rmse", as_pandas=True, seed=123)
    rmse_l1.append(cv_results["test-rmse-mean"].tail(1).values[0])
    
print("Best rmse as a function of l1:")
print(pd.DataFrame(list(zip(l1_params, rmse_l1)), columns=["l1", "rmse"])
```

#### Compare Base Learners in XGBoost
- **Linear Base Learner** : Sum of linear terms exactly as we would find in a linear or logistic regression model.
- When we combine many of these base models into an ensemble we get a weighted sum of linear models, which itself is linear. Since we don't get any non linear combination of features in the final model, this approach is rarely used, as we can get identical performance from a regularized linear model.
- **Tree Base Learner** : The **tree base learner** uses decision trees as base models. When the decision trees are all combined into an ensemble, thier combination becomes an nonlinear function of each individual tree, which itself is nonlinear.

##### Creating dataframes from multiple equal-length lists

```python
pd.DataFrame(list(zip(list1, list2)), columns=['list1', 'list2']))
```

- In python 3, zip creates a generator or an object that doesnt have to be completely instantiated at runtime. Inorder for the zipped pair of list to be instantitated, we have to cast the zip generator object into a list directly.
























































- 

























































