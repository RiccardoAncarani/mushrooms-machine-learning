
# Mushrooms Classifier
## Safe to eat or deadly poison?

Dataset taken from [Kaggle](https://www.kaggle.com/uciml/mushroom-classification)

### Context

Although this dataset was originally contributed to the UCI Machine Learning repository nearly 30 years ago, mushroom hunting (otherwise known as "shrooming") is enjoying new peaks in popularity. Learn which features spell certain death and which are most palatable in this dataset of mushroom characteristics. And how certain can your model be?

### Content

This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy.

+ Time period: Donated to UCI ML 27 April 1987
+ Inspiration

What types of machine learning models perform best on this dataset?
Which features are most indicative of a poisonous mushroom?


### Feature description
Attribute Information: (classes: edible=e, poisonous=p)

+ cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
+ cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
+ cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
+ bruises: bruises=t,no=f
+ odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
+ gill-attachment: attached=a,descending=d,free=f,notched=n
+ gill-spacing: close=c,crowded=w,distant=d
+ gill-size: broad=b,narrow=n
+ gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
+ stalk-shape: enlarging=e,tapering=t
+ stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
+ stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
+ stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
+ stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
+ stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
+ veil-type: partial=p,universal=u
+ veil-color: brown=n,orange=o,white=w,yellow=y
+ ring-number: none=n,one=o,two=t
+ ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
+ spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
+ population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
+ habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d



```python
'''
Importing the foundamental libraries and reading the dataset with Pandas
'''
import pandas as pd
import numpy as np

data = pd.read_csv("mushrooms.csv")
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>p</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>y</td>
      <td>t</td>
      <td>a</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e</td>
      <td>b</td>
      <td>s</td>
      <td>w</td>
      <td>t</td>
      <td>l</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>p</td>
      <td>x</td>
      <td>y</td>
      <td>w</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>g</td>
      <td>f</td>
      <td>n</td>
      <td>f</td>
      <td>w</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>n</td>
      <td>a</td>
      <td>g</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
target = 'class' # The class we want to predict
labels = data[target]

features = data.drop(target, axis=1) # Remove the target class from the dataset

```

### Feature transformation
Since we have only categorical features, we cannot feed them directly into sklearn classifiers.
The technique we are going to use is called **One Hot Encoding** and what it does is basically add a new binary feature for each value the categorical feature has.

In pandas we can use the get_dummies function:


```python
categorical = features.columns # Since every fearure is categorical we use features.columns
features = pd.concat([features, pd.get_dummies(features[categorical])], axis=1) # Convert every categorical feature with one hot encoding
features.drop(categorical, axis=1, inplace=True) # Drop the original feature, leave only the encoded ones

labels = pd.get_dummies(labels)['p'] # Encode the target class, 1 is deadly 0 is safe

```


```python
''' 
Split the dataset into training and testing, the 80% of the records are in the trainig set
'''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.2, random_state=0)
```


```python
'''
Train predict pipeline
'''

from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
   
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    results['train_time'] = end - start
        
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    results['pred_time'] = end - start
            
    results['acc_train'] = accuracy_score(y_train[:300],predictions_train)
        
    results['acc_test'] = accuracy_score(y_test,predictions_test)
    
    results['f_train'] = fbeta_score(y_train[:300],predictions_train, beta=0.5)
        
    results['f_test'] = fbeta_score(y_test,predictions_test, beta=0.5)
       
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
        
    return results
```

## Choosing the best model
We use three different model:
+ Gaussian Naive Bayes
+ Random Forests
+ kNN

The results are stored in the results dictionary:


```python
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

clf_A = GaussianNB()
clf_B = RandomForestClassifier()
clf_C = KNeighborsClassifier()

training_length = len(X_train)
samples_1 = int(training_length * 0.01)
samples_10 = int(training_length * 0.1)
samples_100 = int(training_length * 1)

results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

```

    GaussianNB trained on 64 samples.
    GaussianNB trained on 649 samples.
    GaussianNB trained on 6499 samples.
    RandomForestClassifier trained on 64 samples.
    RandomForestClassifier trained on 649 samples.
    RandomForestClassifier trained on 6499 samples.
    KNeighborsClassifier trained on 64 samples.
    KNeighborsClassifier trained on 649 samples.
    KNeighborsClassifier trained on 6499 samples.


## The best model
The best model is Random Forest classifier, that achieved 100% accuracy with the test set!
We don't even need hype params. tuning:


```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
                verbose=0, warm_start=False)



## Final thoughts
We can now print the most important features:
The results show that the foul odor is the most discriminant feature!


```python
z = zip(clf.feature_importances_,X_train.columns)
z.sort(reverse=True)
z[:10]
```




    [(0.15844522158060251, 'odor_f'),
     (0.072093232716836098, 'gill-size_n'),
     (0.071449650799149014, 'ring-type_p'),
     (0.059524344656014208, 'stalk-surface-below-ring_k'),
     (0.054395896612936, 'gill-color_b'),
     (0.053292416415563093, 'odor_n'),
     (0.051462205469969005, 'stalk-root_e'),
     (0.037758414413626332, 'odor_p'),
     (0.037439645501368912, 'stalk-surface-above-ring_k'),
     (0.033770321762183406, 'odor_c')]




```python

```
