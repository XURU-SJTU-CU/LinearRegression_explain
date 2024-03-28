# LinearRegression_explain

## 源代码
### part 1
```python
!pip install -q sklearn
%tensorflow_version 2.x

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
```
### part1 对应的解释
train.csv 文件中对应的内容如下所示
![image](https://github.com/XURU-SJTU-CU/LinearRegression_explain/assets/33889393/a9884c99-24ec-4265-801e-0338a6d3ea91)

对于pandas的dataFrame中的pop的解释如下

参考链接
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pop.html

```python
df = pd.DataFrame([('falcon','bird',389.0),
                   ('lion', 'mammal', 80.5),
                   ('monkey', 'mammal', np.nan)],
                    columns = ('name','class','max_speed'))

```
 每一行相当于一个tuple，包含各种信息。调用pop函数之后会返回对应的列的信息，pop之后原本的dataFrame中的对应列会被删除。
```python
df.pop('class')
```
```python
# result
0      bird
1    mammal
2    mammal
Name: class, dtype: object
```

```python
>>> df
    name  max_speed
0  falcon      389.0
1  parrot       24.0
2    lion       80.5
3  monkey        NaN
```
### part 2
```python
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)
```
### part 2对应的解释

对应表格中的数据，有一些是能够量化的，比如年纪，船票价格之类的。但是有一些只能进行分类，比如船票仓位头等舱，二等舱之类的。或者是乘客的性别是男或者是女。对于不同种类的feature需要进行不同的处理。

第一个for循环就是在整个列表中寻找某一个分类中对应的所有子类，比如在性别栏中寻找所有类型。进行这样的处理是为了能够将male，female之类的类型映射到数字上，这样就不需要手动进行数据转换。

```python
dftrain.head()
```
可以查看数据表的前几行的内容
```python
dftrain.describe()
dftrain.shape
```
可以对于表格中的数据进行简单的数据分析，比如求平均值，方差之类的操作
```python
y_train.head()
#result
0    0
1    1
2    1
3    1
4    0
Name: survived, dtype: int64
```
```python
dftrain.age.hist(bins=20)
```
上方的命令调用了pyplot画出age对应栏的20个数据的直方图，可以直观地观察表格中的数据。

### part3
```python
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

```

### part3 解释


make_input_fn 输入对应的数据，数据对应的label，迭代的次数，是否打乱数据，以及每一笔数据的大小 

### part 4
```python
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears consoke output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model
```
### part4 解释


### part5
```python
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')

```
