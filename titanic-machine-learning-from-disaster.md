# Titanic - Machine Learning from Disaster

Using a simple logistic regression classifier

### Importing Libraries


```python
import numpy as np
import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImpute

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC
```

    /kaggle/input/titanic/train.csv
    /kaggle/input/titanic/test.csv
    /kaggle/input/titanic/gender_submission.csv


### Load the files


```python
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
```

### Quick glance through the data


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>1305</td>
      <td>3</td>
      <td>Spector, Mr. Woolf</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>A.5. 3236</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1306</td>
      <td>1</td>
      <td>Oliva y Ocana, Dona. Fermina</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17758</td>
      <td>108.9000</td>
      <td>C105</td>
      <td>C</td>
    </tr>
    <tr>
      <th>415</th>
      <td>1307</td>
      <td>3</td>
      <td>Saether, Mr. Simon Sivertsen</td>
      <td>male</td>
      <td>38.5</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/O.Q. 3101262</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>416</th>
      <td>1308</td>
      <td>3</td>
      <td>Ware, Mr. Frederick</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>359309</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1309</td>
      <td>3</td>
      <td>Peter, Master. Michael J</td>
      <td>male</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>2668</td>
      <td>22.3583</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 11 columns</p>
</div>




```python
df_gender.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train.shape
```




    (891, 12)




```python
df_test.sample()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>235</th>
      <td>1127</td>
      <td>3</td>
      <td>Vendel, Mr. Olof Edvin</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>350416</td>
      <td>7.8542</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
df_test.shape
```




    (418, 11)



### Checking for missing data


```python
msno.matrix(df_train)

```




    <AxesSubplot:>




    
![png](titanic-machine-learning-from-disaster_files/titanic-machine-learning-from-disaster_14_1.png)
    


There are some nulls in Age, Cabin adn Embarked columns


```python
df_train.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
df_test.isnull().sum()
```




    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64



### Analysing and Visualising Data


```python
women = df_train.loc[df_train.Sex == 'female']["Survived"]
rate_women = round(sum(women)*100/len(women),2)

men = df_train.loc[df_train.Sex == 'male']["Survived"]
rate_men = round(sum(men)*100/len(men),2)

gender_ratio = {'Men': rate_men, 'Women': rate_women}
df_gender_ratio = pd.DataFrame(gender_ratio.items(), columns = ['Gender', 'Survival Rate'])
df_gender_ratio
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Survival Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Men</td>
      <td>18.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Women</td>
      <td>74.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
palette = {'Men': 'red', 'Women':'salmon'}
sns.barplot(x=df_gender_ratio['Gender'], y=df_gender_ratio['Survival Rate'],palette=palette )

plt.title('Survival % per Gender', fontweight = 'bold')
plt.xlabel('Gender',fontweight = 'bold')
plt.ylabel('Survival Rate',fontweight = 'bold')
plt.show()
```


    
![png](titanic-machine-learning-from-disaster_files/titanic-machine-learning-from-disaster_20_0.png)
    



```python
class_1 = df_train.loc[df_train.Pclass == 1]["Survived"]
class_1_rate = round(sum(class_1)*100/len(class_1),2)

class_2 = df_train.loc[df_train.Pclass == 2]["Survived"]
class_2_rate = round(sum(class_2)*100/len(class_2),2)

class_3 = df_train.loc[df_train.Pclass == 3]["Survived"]
class_3_rate = round(sum(class_3)*100/len(class_3),2)

class_rates = {'class_1': class_1_rate, 'class_2': class_2_rate, 'class_3': class_3_rate}
df_class_rates = pd.DataFrame(class_rates.items(), columns=['PClass', 'Survival Rate'])
df_class_rates
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PClass</th>
      <th>Survival Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>class_1</td>
      <td>62.96</td>
    </tr>
    <tr>
      <th>1</th>
      <td>class_2</td>
      <td>47.28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>class_3</td>
      <td>24.24</td>
    </tr>
  </tbody>
</table>
</div>




```python
palette = {'class_1': 'green', 'class_2': 'salmon','class_3':'red'}
sns.barplot(x=df_class_rates['PClass'], y=df_class_rates['Survival Rate'], palette=palette)

plt.title('Survival % per Class', fontweight = 'bold')
plt.xlabel('PClass',fontweight = 'bold')
plt.ylabel('Survival Rate',fontweight = 'bold')
plt.show()
```


    
![png](titanic-machine-learning-from-disaster_files/titanic-machine-learning-from-disaster_22_0.png)
    



```python
sns.set_palette("RdBu")

sample = df_train
g = sns.pairplot(sample, diag_kind='kde', 
                 plot_kws={'alpha': 0.50, 's': 50, 'edgecolor': 'k'},
                 height=5, hue='Survived')
# g.map_diag(sns.kdeplot, shade=True)
# g.map_upper(plt.scatter, alpha=0.5)
# g.map_lower(sns.kdeplot, shade=False, shade_lowest=False, cbar=True)

plt.tight_layout()
plt.show()
```


    
![png](titanic-machine-learning-from-disaster_files/titanic-machine-learning-from-disaster_23_0.png)
    


### Transform Data


```python
# For ease of analysis, we are creating a new column 'Gender' which is integer. Contains 0 for male and 1 for female.
df_train['Gender'] = df_train['Sex'].replace({'male': 0, 'female': 1}).astype(int)
df_test['Gender'] = df_test['Sex'].replace({'male': 0, 'female': 1}).astype(int)

```


```python
# Create a SimpleImputer object to impute missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the 'age' column and transform the values
df_train['Age'] = imputer.fit_transform(df_train[['Age']])
df_test['Age'] = imputer.transform(df_test[['Age']])
```

### Build Logistic Regression Classifier

Separate the predictors and response for train and test subsets


```python
X_train, y_train = df_train[['Pclass', 'Gender','Age','SibSp','Parch']], df_train['Survived']
X_test = df_test[['Pclass', 'Gender','Age','SibSp','Parch']]
```

Train the model and use it for predictions


```python
classifier = LogisticRegression(solver='newton-cg')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

df_test["Survived"] = classifier.predict(X_test)
predicted_results = df_test[["PassengerId", "Survived"]]
```


```python
predicted_results.shape
```




    (418, 2)




```python
predicted_results.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>897</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>898</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>899</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>900</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>901</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Let us visualise the model's performance


```python
fig, axes = plt.subplots(3, 1)

model = classifier
visualgrid = [
    FeatureImportances(model, ax=axes[0]),
    ConfusionMatrix(model, ax=axes[1]),
    ClassificationReport(model, ax=axes[2]),
]


for viz in visualgrid:
    viz.fit(X_train, y_train)
    viz.score(X_test, df_test["Survived"])
    viz.finalize()
    
plt.suptitle(r'\textbf{Logistic regression model diagnostics}');
plt.tight_layout();
plt.show() 
```

    /opt/conda/lib/python3.7/site-packages/sklearn/base.py:451: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
      "X does not have valid feature names, but"
    /opt/conda/lib/python3.7/site-packages/yellowbrick/model_selection/importances.py:199: YellowbrickWarning: detected multi-dimensional feature importances but stack=False, using mean to aggregate them.
      YellowbrickWarning,
    /opt/conda/lib/python3.7/site-packages/sklearn/base.py:451: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
      "X does not have valid feature names, but"
    /opt/conda/lib/python3.7/site-packages/sklearn/base.py:451: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
      "X does not have valid feature names, but"



    
![png](titanic-machine-learning-from-disaster_files/titanic-machine-learning-from-disaster_35_1.png)
    


From these results, it looks like the model is doing a pretty good job at the predicting the results.

### Convert Result to A File


```python
predicted_results.to_csv('/kaggle/working/PredictedResults.csv', index = False)
```
