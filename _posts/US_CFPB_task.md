```python
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
```

# US Consumer FInancial Protection Bureau

As an intermediary, the CFPB receives a large number of complaints. To help the CFPB better manage them, they would like to reduce the complaints logged to those that are most likely to be successful. To help them to do this, they would like to be able to indicate to a consumer whether it is likely that their complaint will be accepted. This will help the consumer to decide whether they wish to go through the complaints process, and will reduce the number of complaints logged.

Therefore, your objective is to build a classification model that can be used to:

   1. Determine whether a consumer's complaint will be accepted and whether they are likely to receive relief.
   2. Help the CFPB understand what factors affect how a company responds to the complaints that it receives.
   
This investigation is split into three sections:

### [Exploratory Data Analysis](#eda)

### [Classification](#clf)

### [Conclusion](#fin)

On my [github](www.github.com/tmerrittsmith) page, I have another notebook where I tried some neural net approaches. However, the performance wasn't as good as Logistic Regression used here.

<a id='eda'></a>

## Exploratory Data Analysis

In this section, we perform some basic data cleaning, and plot the data to understand some of the features.


```python
## if you haven't got the data, you can download it from here
## I've inluded a sample in this repo, just so you can see the code running
```


```python
df = pd.read_csv('Consumer_complaints.csv')
```


```python
df.columns = [c.replace(' ','_').lower().replace('-','').replace('?','') for c in df.columns]
df.columns
```




    Index(['date_received', 'product', 'subproduct', 'issue', 'subissue',
           'consumer_complaint_narrative', 'company_public_response', 'company',
           'state', 'zip_code', 'tags', 'consumer_consent_provided',
           'submitted_via', 'date_sent_to_company', 'company_response_to_consumer',
           'timely_response', 'consumer_disputed', 'complaint_id'],
          dtype='object')




```python
def extract_year_month_date(datetime_series, dateformat=None, is_string=False):
    ## take a datetime series and return a dataframe containing the year, month, 
    ## day and dayofweek
    if dateformat is None:
        infer_date_format = True
    else:
        infer_date_format = False
    
    if is_string:
        datetime_series = pd.to_datetime(datetime_series, infer_datetime_format=infer_date_format, format=dateformat)
#         print(datetime_series)
    colname = datetime_series.name
    year = datetime_series.apply(lambda x: x.year).to_frame().\
                           rename(columns = {colname:'year_{}'.format(colname)})
    month = datetime_series.apply(lambda x: x.month).to_frame().\
                            rename(columns = {colname:'month_{}'.format(colname)})
    day = datetime_series.apply(lambda x: x.day).to_frame().\
                          rename(columns = {colname:'day_{}'.format(colname)})
    dayofweek = datetime_series.apply(lambda x: x.dayofweek).to_frame().\
                                rename(columns = {colname:'dayofweek_{}'.format(colname)})
    return year.join(month).join(day).join(dayofweek).astype(int)

df = df.join(extract_year_month_date(df['date_received'], is_string=True, dateformat="%d/%M/%Y"))
df = df.join(extract_year_month_date(df['date_sent_to_company'], is_string=True, dateformat="%d/%M/%Y"))
```


```python
# Look at the total complaints for different products, by year 
pd.crosstab(df['product'], df.year_date_received)
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
      <th>year_date_received</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
    </tr>
    <tr>
      <th>product</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bank account or service</th>
      <td>0</td>
      <td>12212</td>
      <td>13388</td>
      <td>14662</td>
      <td>17140</td>
      <td>21849</td>
      <td>6956</td>
    </tr>
    <tr>
      <th>Checking or savings account</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9947</td>
    </tr>
    <tr>
      <th>Consumer Loan</th>
      <td>0</td>
      <td>1986</td>
      <td>3117</td>
      <td>5457</td>
      <td>7888</td>
      <td>9602</td>
      <td>3558</td>
    </tr>
    <tr>
      <th>Credit card</th>
      <td>1260</td>
      <td>15353</td>
      <td>13105</td>
      <td>13974</td>
      <td>17300</td>
      <td>21066</td>
      <td>7132</td>
    </tr>
    <tr>
      <th>Credit card or prepaid card</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11921</td>
    </tr>
    <tr>
      <th>Credit reporting</th>
      <td>0</td>
      <td>1873</td>
      <td>14380</td>
      <td>29239</td>
      <td>34273</td>
      <td>44081</td>
      <td>16578</td>
    </tr>
    <tr>
      <th>Credit reporting, credit repair services, or other personal consumer reports</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>59186</td>
    </tr>
    <tr>
      <th>Debt collection</th>
      <td>0</td>
      <td>0</td>
      <td>11069</td>
      <td>39148</td>
      <td>39757</td>
      <td>40492</td>
      <td>41101</td>
    </tr>
    <tr>
      <th>Money transfer, virtual currency, or money service</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2213</td>
    </tr>
    <tr>
      <th>Money transfers</th>
      <td>0</td>
      <td>0</td>
      <td>559</td>
      <td>1169</td>
      <td>1619</td>
      <td>1567</td>
      <td>440</td>
    </tr>
    <tr>
      <th>Mortgage</th>
      <td>1276</td>
      <td>38109</td>
      <td>49401</td>
      <td>42962</td>
      <td>42353</td>
      <td>41471</td>
      <td>26622</td>
    </tr>
    <tr>
      <th>Other financial service</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>116</td>
      <td>312</td>
      <td>466</td>
      <td>165</td>
    </tr>
    <tr>
      <th>Payday loan</th>
      <td>0</td>
      <td>0</td>
      <td>194</td>
      <td>1706</td>
      <td>1586</td>
      <td>1567</td>
      <td>493</td>
    </tr>
    <tr>
      <th>Payday loan, title loan, or personal loan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2245</td>
    </tr>
    <tr>
      <th>Prepaid card</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>336</td>
      <td>1784</td>
      <td>1250</td>
      <td>449</td>
    </tr>
    <tr>
      <th>Student loan</th>
      <td>0</td>
      <td>2840</td>
      <td>3005</td>
      <td>4283</td>
      <td>4501</td>
      <td>8087</td>
      <td>15896</td>
    </tr>
    <tr>
      <th>Vehicle loan or lease</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2873</td>
    </tr>
    <tr>
      <th>Virtual currency</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>7</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
## looking at the list of products, we can see that there's some unification which seems reasonable.

remap_products = {
     # product:remapped_product
    'Credit card':'Credit card or prepaid card',
    'Credit reporting, credit repair services, or other personal consumer reports':'Credit reporting',
    'Money transfers':'Money transfer, virtual currency, or money service',
    'Virtual currency':'Money transfer, virtual currency, or money service',
    'Payday loan, title loan, or personal loan':'Payday loan',
    'Prepaid card':'Credit card or prepaid card', 
    
}

df['product'] = df['product'].apply(lambda x: remap_products.get(x) if remap_products.get(x) else x)
```


```python
# unify the relief responses
relief = ['Closed with monetary relief',
             'Closed with non-monetary relief',
             'Closed with relief']

df['relief_received'] = df.company_response_to_consumer.apply(lambda x:1 if x in relief else 0)
```


```python
fig, ax = plt.subplots(figsize=(10,10))
tab = pd.crosstab(df['product'],df['relief_received'])
(tab.T / tab.sum(axis=1)).T.plot(kind='bar', ax=ax)
plt.title('Percentage company responses for each product type.')
# plt.tight_layout()

plt.show()
```


![png](/assets/images/us_cfpb/output_11_0.png)



```python
fig, ax = plt.subplots(figsize=(20,10))

df.groupby(['year_date_received','product']).relief_received.mean().unstack().plot(ax=ax)
plt.legend(loc='lower center', ncol=2)
plt.suptitle('Annual relief rate by product')
plt.show()
```


![png](/assets/images/us_cfpb/output_12_0.png)


#### Company size
We observe that most companies have a very small number of complaints, while others have a very large number. Does this affect the relief rate?


```python
print(\
      "{0:.1f}% of companies do less than 100 transactions.".format(\
      (df.groupby('company').count()['product'] < 100).sum() /\
                            df.company.nunique() * 100))
```

    89.4% of companies do less than 100 transactions.



```python
# we note that most companies are very small
company_size = df.groupby('company').count()['product']
company_size_bin = pd.cut(company_size, bins=[0, 100, 1000, 10000, 50000, 100000],
                          labels=['very_small', 'small', 'medium', 'large', 'very_large']).reset_index()
company_size_bin.columns = ['company', 'company_size']
df = df.merge(company_size_bin, on='company')
```


```python
df['company_relief_rate'] = df.groupby('company').relief_received.transform('mean')

fig, ax = plt.subplots(figsize=(10,10))
df.boxplot(column='company_relief_rate', by='company_size',ax=ax)
plt.show()
```


![png](/assets/images/us_cfpb/output_16_0.png)



```python
# We can see that very large companies receive most complaints about bank accounts, credit reporting, and mortgages
pd.crosstab(df['product'], df['company_size'])
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
      <th>company_size</th>
      <th>very_small</th>
      <th>small</th>
      <th>medium</th>
      <th>large</th>
      <th>very_large</th>
    </tr>
    <tr>
      <th>product</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bank account or service</th>
      <td>1000</td>
      <td>6115</td>
      <td>30471</td>
      <td>21322</td>
      <td>27299</td>
    </tr>
    <tr>
      <th>Checking or savings account</th>
      <td>169</td>
      <td>655</td>
      <td>3625</td>
      <td>2559</td>
      <td>2939</td>
    </tr>
    <tr>
      <th>Consumer Loan</th>
      <td>2792</td>
      <td>9735</td>
      <td>11308</td>
      <td>4441</td>
      <td>3332</td>
    </tr>
    <tr>
      <th>Credit card or prepaid card</th>
      <td>572</td>
      <td>3801</td>
      <td>27592</td>
      <td>58072</td>
      <td>14893</td>
    </tr>
    <tr>
      <th>Credit reporting</th>
      <td>1942</td>
      <td>5515</td>
      <td>5512</td>
      <td>3934</td>
      <td>182707</td>
    </tr>
    <tr>
      <th>Debt collection</th>
      <td>36514</td>
      <td>58364</td>
      <td>57402</td>
      <td>15051</td>
      <td>4236</td>
    </tr>
    <tr>
      <th>Money transfer, virtual currency, or money service</th>
      <td>395</td>
      <td>831</td>
      <td>4960</td>
      <td>635</td>
      <td>764</td>
    </tr>
    <tr>
      <th>Mortgage</th>
      <td>7786</td>
      <td>15079</td>
      <td>58160</td>
      <td>87537</td>
      <td>73632</td>
    </tr>
    <tr>
      <th>Other financial service</th>
      <td>324</td>
      <td>182</td>
      <td>263</td>
      <td>149</td>
      <td>141</td>
    </tr>
    <tr>
      <th>Payday loan</th>
      <td>1533</td>
      <td>4194</td>
      <td>1647</td>
      <td>216</td>
      <td>201</td>
    </tr>
    <tr>
      <th>Student loan</th>
      <td>1640</td>
      <td>4858</td>
      <td>9704</td>
      <td>20787</td>
      <td>1623</td>
    </tr>
    <tr>
      <th>Vehicle loan or lease</th>
      <td>278</td>
      <td>938</td>
      <td>1128</td>
      <td>293</td>
      <td>236</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(10,10))
df.groupby(['year_date_received','company_size']).relief_received.mean().unstack().plot(ax=ax)
plt.title('Proportion of complaints in which relief is given \nfor companies of different size, by year')
plt.show()
```


![png](/assets/images/us_cfpb/output_18_0.png)


## Discussion of exploratory data analysis

From the exploratory analysis above, we can make a few initial observations:
- Credit cards and prepaid cards, credit reporting, and account services have the highest relief rates, but these are gradually decreasing year-on-year
- Larger companies have a higher average relief rate, but there is much greater variation in small companies.

<a id='clf'></a>

# Classification

In this task, we are asked to build a classification model that can be used to:

   1. Determine whether a consumer's complaint will be accepted and whether they are likely to receive relief.
   2. Help the CFPB understand what factors affect how a company responds to the complaints that it receives.
    
From this statement, we can begin to make plans about what sort of model to use in classification:
- an interpretable model will be helpful as we will immediately gain an insight into factors affecting company response
- if the model is not interpretable, we will need to use some kind of model inspection or explanation technique
- Relevant factors may be found in the text, but also in the metadata of the complaint (product, issue, tag etc.) so we should also include these features in our model.

#### Baseline
In any classification problem, it's useful to establish a baseline for further development, and for nlp problems this is usually a bag-of-words input with a linear model - here we will try Logistic Regression, and a Linear SVC. 


#### Data preprocessing
It is standard practice to undertake some cleaning of the text, such as removing stopwords and punctuation; from a brief glance at some samples, we can see that the complaints are anonymised using 'xxxx' tokens - these should also be removed. As mentioned above, a bag-of-words model is a standard approach, but we can also use Tf-Idf to give a relevance weighting to tokens within the text. 

The following metadata features will also be included: product, issue, sub-issue, company size and tag. We will not include the actual company, as this may lead to overfitting based on the relief rate of companies in the training set, and would not generalise to new companies as they appear in the data.

There is a 4:1 imbalance in the data, which is not too extreme, but should be accounted for. As a minimum, the data must be stratified in the train-test split, and during model development we can explore the use of class weights and oversampling of the data.



```python
df.dropna(subset=['consumer_complaint_narrative'])\
  .relief_received.value_counts() / df.dropna(subset=['consumer_complaint_narrative']).shape[0]
```




    0    0.819553
    1    0.180447
    Name: relief_received, dtype: float64




```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import scipy.stats as stats
np.random.seed(31415)

```


```python
df_text = df.copy()
df_text.dropna(axis=0,subset=['consumer_complaint_narrative'], inplace=True)
df_text.shape
X = df_text['consumer_complaint_narrative'].str.replace('xx+','')
y = df_text.relief_received
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=31415)

```


```python
# baseline model for logistic regression and LinearSVC
relief_tags = ['Closed with non-monetary relief',
       'Closed with monetary relief','Closed with relief',]


vectorizer = CountVectorizer(strip_accents = 'unicode',
                                stop_words = 'english',
                                lowercase = True,
                                max_df = 0.5, 
                                min_df = 100)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

models = {'lr':LogisticRegression(class_weight={0:1,1:5}, random_state=31415), 
          'svc':LinearSVC(class_weight={0:1,1:5}, random_state=31415)}
preds = {}

for name, model in models.items():
    print(name)
    model.fit(X_train, y_train)
    preds[name] = model.predict(X_test)
    print(classification_report(y_test, preds[name]))

```

    lr


    C:\Users\thomas.merritt-smith\AppData\Local\Continuum\anaconda3\envs\uscfpb_test\lib\site-packages\sklearn\linear_model\_logistic.py:939: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html.
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)


                  precision    recall  f1-score   support
    
               0       0.91      0.66      0.77     32777
               1       0.31      0.69      0.43      7217
    
        accuracy                           0.67     39994
       macro avg       0.61      0.68      0.60     39994
    weighted avg       0.80      0.67      0.71     39994
    
    svc
                  precision    recall  f1-score   support
    
               0       0.89      0.74      0.81     32777
               1       0.33      0.59      0.42      7217
    
        accuracy                           0.71     39994
       macro avg       0.61      0.66      0.62     39994
    weighted avg       0.79      0.71      0.74     39994
    


    C:\Users\thomas.merritt-smith\AppData\Local\Continuum\anaconda3\envs\uscfpb_test\lib\site-packages\sklearn\svm\_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


#### Leveraging additional meta-data


```python
from scipy.sparse import hstack

ohe = OneHotEncoder(handle_unknown='ignore')
# we won't use company name, as this would lead to overfitting to individual companies
features = ['product','issue','subissue','company_size'] 

X = df_text[['product','issue','subissue','company_size','consumer_complaint_narrative']]
X[['product','issue','subissue']] = X[['product','issue','subissue']].fillna('none')
y = df_text.relief_received
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=31415)

vectorizer_meta_data = CountVectorizer()
X_train_text = vectorizer_meta_data.fit_transform(X_train['consumer_complaint_narrative'])
X_test_text = vectorizer_meta_data.transform(X_test['consumer_complaint_narrative'])
X_train_features = ohe.fit_transform(X_train[features],)
X_test_features = ohe.transform(X_test[features])

X_train = hstack([X_train_text, X_train_features])
X_test = hstack([X_test_text, X_test_features])



```

    C:\Users\thomas.merritt-smith\AppData\Local\Continuum\anaconda3\envs\uscfpb_test\lib\site-packages\pandas\core\frame.py:3509: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self[k1] = value[k2]



```python
model = LogisticRegression(class_weight='balanced', random_state=31415)
model.fit(X_train, y_train)
```

    C:\Users\thomas.merritt-smith\AppData\Local\Continuum\anaconda3\envs\uscfpb_test\lib\site-packages\sklearn\linear_model\_logistic.py:939: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html.
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)





    LogisticRegression(C=1.0, class_weight='balanced', dual=False,
                       fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                       max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=31415, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
preds = model.predict(X_test)
print(classification_report(y_test, preds))
```

                  precision    recall  f1-score   support
    
               0       0.91      0.72      0.80     32777
               1       0.34      0.66      0.45      7217
    
        accuracy                           0.71     39994
       macro avg       0.62      0.69      0.63     39994
    weighted avg       0.80      0.71      0.74     39994
    


#### Inspecting the Logistic Regression model


```python
feature_melt = []
for feature, categories in zip(features,ohe.categories_):
    melted = [feature] * len(categories)
    feature_melt += melted

categories = np.hstack(ohe.categories_)

coef_df = pd.DataFrame([categories, model.coef_[:,-len(categories):][0], feature_melt]).T
coef_df.columns = ['category', 'coef', 'original_feature']

fig, ax = plt.subplots(figsize=(10,10))
coef_df.boxplot(column='coef', 
           by='original_feature', 
           ax=ax,
           )
plt.suptitle('Boxplot of coefficients for meta-data features of complaints')
plt.show()
```


![png](/assets/images/us_cfpb/output_31_0.png)



```python
# Display the coefficients for products
coef_df[coef_df.original_feature == 'product'].sort_values('coef', ascending=False)
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
      <th>category</th>
      <th>coef</th>
      <th>original_feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Credit card or prepaid card</td>
      <td>0.545969</td>
      <td>product</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Bank account or service</td>
      <td>0.326113</td>
      <td>product</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Checking or savings account</td>
      <td>0.225128</td>
      <td>product</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Debt collection</td>
      <td>0.0556301</td>
      <td>product</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Other financial service</td>
      <td>0.0510896</td>
      <td>product</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Consumer Loan</td>
      <td>0.0438741</td>
      <td>product</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Vehicle loan or lease</td>
      <td>-0.001235</td>
      <td>product</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Credit reporting</td>
      <td>-0.0166776</td>
      <td>product</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Money transfer, virtual currency, or money ser...</td>
      <td>-0.156185</td>
      <td>product</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Payday loan</td>
      <td>-0.386861</td>
      <td>product</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Mortgage</td>
      <td>-0.526208</td>
      <td>product</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Student loan</td>
      <td>-0.649351</td>
      <td>product</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Display the coefficients for company size
coef_df[coef_df.original_feature == 'company_size'].sort_values('coef', ascending=False)
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
      <th>category</th>
      <th>coef</th>
      <th>original_feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>381</th>
      <td>very_large</td>
      <td>0.463117</td>
      <td>company_size</td>
    </tr>
    <tr>
      <th>379</th>
      <td>medium</td>
      <td>0.136443</td>
      <td>company_size</td>
    </tr>
    <tr>
      <th>378</th>
      <td>large</td>
      <td>-0.073706</td>
      <td>company_size</td>
    </tr>
    <tr>
      <th>382</th>
      <td>very_small</td>
      <td>-0.471874</td>
      <td>company_size</td>
    </tr>
    <tr>
      <th>380</th>
      <td>small</td>
      <td>-0.542694</td>
      <td>company_size</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Display the coefficients for issues
coef_df[coef_df.original_feature == 'issue'].sort_values('coef', ascending=False)[:10]
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
      <th>category</th>
      <th>coef</th>
      <th>original_feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>156</th>
      <td>Unable to get credit report/credit score</td>
      <td>0.875798</td>
      <td>issue</td>
    </tr>
    <tr>
      <th>134</th>
      <td>Problems caused by my funds being low</td>
      <td>0.542986</td>
      <td>issue</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Late fee</td>
      <td>0.3429</td>
      <td>issue</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Communication tactics</td>
      <td>0.326751</td>
      <td>issue</td>
    </tr>
    <tr>
      <th>159</th>
      <td>Unauthorized transactions/trans. issues</td>
      <td>0.305247</td>
      <td>issue</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Improper contact or sharing of info</td>
      <td>0.284458</td>
      <td>issue</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Managing, opening, or closing account</td>
      <td>0.250215</td>
      <td>issue</td>
    </tr>
    <tr>
      <th>108</th>
      <td>Other fee</td>
      <td>0.233449</td>
      <td>issue</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Billing disputes</td>
      <td>0.198935</td>
      <td>issue</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Credit monitoring or identity protection</td>
      <td>0.177035</td>
      <td>issue</td>
    </tr>
  </tbody>
</table>
</div>




```python
words = vectorizer.get_feature_names()
word_coefs = pd.DataFrame([words, models['lr'].coef_[0]]).T
word_coefs.columns = ['words','coef']

```


```python
# Display the words with largest positive coefficients
print(word_coefs.sort_values('coef', ascending=False)[:60])
```

                       words      coef
    2844           jefferson   2.72214
    1829             dynamic   2.29037
    4377            rushcard   1.91164
    1920            enhanced   1.57078
    2789          interstate    1.4535
    4376                rush    1.3884
    3837       professionals  0.830137
    1096            citigold  0.801865
    5132           universal  0.792687
    562   annualcreditreport  0.776771
    3585            partners   0.68735
    520               allied  0.623358
    1019                cbna  0.613705
    2586           hurricane   0.58328
    1265                conn  0.560628
    1256         conflicting  0.557217
    4885          technology  0.530956
    4185              repaye  0.526669
    2435               guide  0.525563
    1037       certification  0.522281
    3318               multi  0.518445
    281                  809  0.499301
    548         amortization  0.497155
    3485          operations  0.495929
    1212          completing  0.493841
    715             attitude  0.491621
    1016             cavalry   0.48636
    4607              solely  0.483301
    3075                  lt  0.473938
    1957                 erc  0.472124
    1672            diligent  0.471448
    1811             driving  0.471193
    2821                  iq  0.467393
    5048           triggered  0.464511
    221                  580  0.458511
    5210               usury  0.456739
    1024              ceased  0.452705
    1165            combined  0.452562
    2866        jurisdiction  0.451061
    3869            proposed  0.447958
    2519               hipaa  0.446082
    4331               rings  0.440639
    5259                 vet  0.439018
    4775          subsection  0.435682
    4749              strict  0.434813
    3480             operate  0.429676
    40                 1681b  0.429669
    3088                macy  0.427806
    3218             midland  0.424827
    4313           reversing  0.423636
    689          association  0.422513
    5321              warned  0.421285
    3027              loaded  0.420755
    2895                  ky  0.416796
    4738        straightened  0.416616
    581              apology  0.409819
    2038            existent  0.409039
    1630           desperate  0.408801
    3907          punishment    0.4077
    433       administration  0.406148



```python
import re

def inspect_complaints(string, complaints_array, n=20):

        complaints = complaints_array[complaints_array.apply(\
                                     lambda x:True if re.search(string + '\s',x) else False)]
        
        if n > complaints.shape[0]:
            n = complaints.shape[0]
            print('n was too large, printing all available complaints.\n\n')
        for i in range(n):
            complaint = complaints.iloc[i]
            complaint = complaint.replace(string, '[[' + string + ']]')
            print(complaint)
            print('\n\n\n')
            
inspect_complaints('1681b', X.consumer_complaint_narrative)
```

    While checking my personal credit report, I discovered an Unauthorized and Fraudulent credit inquiry made without my KNOWLEDGE or CONSENT by XXXX on or about XXXX/XXXX/2014 on TRANSUNION credit file. I did not authorized or give permission to anyone employed by this company to make any inquiry and view my credit report. XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. I am requesting that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus and have them remove the unauthorized and fraudulent hard inquiry immediately.
    
    
    
    
    While checking my personal credit report, I noticed an unauthorized and fraudulent credit inquiry made by XXXX on or about XX/XX/XXXX on Transunion. I did not authorized anyone employed by this company to make any inquiry and view my credit report. XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. 
    I have requested that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus immediately and have them remove the unauthorized and fraudulent hard inquiry immediately. I also requested that they remove my personal information from their records. My Social Security # is XXXX and my Date of Birth is XX/XX/XXXX in case it is needed to locate the fraudulent inquiry in their system.
    
    
    
    
    While checking my personal credit report, I discovered an Unauthorized and Fraudulent credit inquiry made without my KNOWLEDGE or CONSENT by XXXX on or about XXXX/XXXX/2015 on TRANSUNION credit file. I did not authorized or give permission to anyone employed by this company to make any inquiry and view my credit report. XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. I am requesting that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus and have them remove the unauthorized and fraudulent hard inquiry immediately.
    
    
    
    
    While checking my personal credit report, I noticed an unauthorized and fraudulent credit inquiry made by XX/XX/XXXX on or about XX/XX/XXXXon Transunion. I did not authorized anyone employed by this company to make any inquiry and view my credit report. XX/XX/XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. 
    I have requested that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus immediately and have them remove the unauthorized and fraudulent hard inquiry immediately. I also requested that they remove my personal information from their records. My Social Security # is XXXX and my Date of Birth is XX/XX/XXXX in case it is needed to locate the fraudulent inquiry in their system.
    
    
    
    
    While checking my personal credit report, I discovered an Unauthorized and Fraudulent credit inquiry made without my KNOWLEDGE or CONSENT by XXXX on or about XXXX/XXXX/2014 on TRANSUNION credit file. I did not authorized or give permission to anyone employed by this company to make any inquiry and view my credit report. XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. I am requesting that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus and have them remove the unauthorized and fraudulent hard inquiry immediately.
    
    
    
    
    While checking my personal credit report, I noticed an unauthorized and fraudulent credit inquiry made by XXXX on or about XX/XX/XXXX on Transunion. I did not authorized anyone employed by this company to make any inquiry and view my credit report. XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. 
    I have requested that they mail me a copy of my signed authorization form that gave them the right to view my credit within XXXX ( XXXX ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus immediately and have them remove the unauthorized and fraudulent hard inquiry immediately. I also requested that they remove my personal information from their records. My Social Security # is XXXX and my Date of Birth is XX/XX/XXXX in case it is needed to locate the fraudulent inquiry in their system.
    
    
    
    
    While checking my personal credit report, I discovered an Unauthorized and Fraudulent credit inquiry made without my KNOWLEDGE or CONSENT by XXXX on or about XXXX/XXXX/2014 on TRANSUNION credit file. I did not authorized or give permission to anyone employed by this company to make any inquiry and view my credit report. XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. I am requesting that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus and have them remove the unauthorized and fraudulent hard inquiry immediately.
    
    
    
    
    While checking my personal credit report, I noticed an unauthorized and fraudulent credit inquiry made by XXXX on or about XXXX/XXXX/XXXXon Transunion. I did not authorized anyone employed by this company to make any inquiry and view my credit report. XXXX XXXX XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. 
    I have requested that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus immediately and have them remove the unauthorized and fraudulent hard inquiry immediately. I also requested that they remove my personal information from their records. My Social Security # is XXXX and my Date of Birth is XX/XX/XXXX in case it is needed to locate the fraudulent inquiry in their system.
    
    
    
    
    While checking my personal credit report, I discovered an Unauthorized and Fraudulent credit inquiry made without my KNOWLEDGE or CONSENT by XXXX on or about XXXX/XXXX/2014 on TRANSUNION credit file. I did not authorized or give permission to anyone employed by this company to make any inquiry and view my credit report. XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. I am requesting that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus and have them remove the unauthorized and fraudulent hard inquiry immediately.
    
    
    
    
    I just checked my personal credit report, just discovered an unauthorized and fraudulent credit inquiry made by XXXX on or about XXXX/XXXX/14 on TRANSUNION. I did not authorized anyone employed by this company or at this company or TRANSUNION to make any inquiry or inquiries and view or show my credit report to anyone, person, company, entity, business, co, corp or similar. XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. I have requested that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof Bearing my Signature or Request in Writing that I authorized them to view my credit report, then I am demanding that TRANSUNION remove the unauthorized and fraudulent hard credit inquiry immediately from my TRANSUNION credit file.
    
    
    
    
    While checking my personal credit report, I noticed an unauthorized and fraudulent credit inquiry made by XXXX on or about XX/XX/XXXX on Transunion. I did not authorized anyone employed by this company to make any inquiry and view my credit report. XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. 
    I have requested that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus immediately and have them remove the unauthorized and fraudulent hard inquiry immediately. I also requested that they remove my personal information from their records. My Social Security # is XXXX and my Date of Birth is XX/XX/XXXX in case it is needed to locate the fraudulent inquiry in their system.
    
    
    
    
    I recently check my Transunion report and it shows an unauthorized Credit Inquiry made from XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX, UT XXXX ( XXXX ) XXXX. 
    
    While checking my personal credit report, which I acquired from [ Transunion ] noticed an inquiry made by the company on XX/XX/2017 I did not authorized anyone employed by the company to make an inquiry and view my credit report. it have violated the Fair Credit Reporting Act Section [[1681b]] ( c ) .You are not legally entitled to make the inquiry. This is a serious breach of my privacy rights. 
    
    I request that you either mail me a copy of my signed authorization form that gave you the right to view my credit within five ( 30 ) business daysso that I can verify its validity
    
    
    
    
    XXXX   XXXX  offered me a chance to receive a secured credit card with an opening deposit twice and I was denied both times. I was n't planning on applying for anything during those times as I was repairing my credit. I applied because I was allegedly pre-approved. That 's not fair to the consumer. Neither  XXXX   XXXX  nor the  XXXX  credit bureaus ( TransUnion,  XXXX , and   XXXX    ) will ho nor removing the inquiries even though I was solicited by   XXXX   XXXX  . The FACTS states inquiries can only be pulled under certain conditions and " firm '' offers a re one of those conditions. Here is a reference to the code under ( c ) ( 1 ) ( B ) ( I ) : 15 U.S. Code [[1681b]] - Permissible purposes of consumer reports ( c ) Furnishing reports in connection with credit or insurance transactions that are not initiated by consumer ( 1 ) In general A consumer reporting agency may furnish a consumer report relating to any consumer pursuant to subparagraph ( A ) or ( C ) of subsection ( a ) ( 3 ) in connection with any credit or insurance transaction that is not initiated by the consumer only if ( A ) the consumer authorizes the agency to provide such report to such person ; or ( B ) ( i ) the transaction consists of a firm offer of credit or insurance ; ( ii ) the consumer reporting agency has complied with subsection ( e ) ; ( iii ) there is not in effect an election by the consumer, made in accor  dance with subsection ( e ), to have the consumers name and address excluded from lists of names provided by the agency pursuant to this paragraph ; and ( iv ) the consumer report does not contain a date of birth that shows that the consumer has not attained the age of 21, or, if the date of birth on the consumer report shows that the consumer has not attained the age of 21, suc h consumer consents to the consumer reporting agency to such furnishing.
    
    
    
    
    While checking my personal credit report, I noticed an unauthorized and fraudulent credit inquiry made by XXXX on or about XX/XX/XXXX on Transunion. I did not authorized anyone employed by this company to make any inquiry and view my credit report. XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. 
    I have requested that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus immediately and have them remove the unauthorized and fraudulent hard inquiry immediately. I also requested that they remove my personal information from their records. My Social Security # is XXXX and my Date of Birth is XX/XX/XXXX in case it is needed to locate the fraudulent inquiry in their system.
    
    
    
    
    Please check the attached FTC ID Theft report, there are several inquiries on my credit report, they are all listed accordingly on that report. I did not give your company permission to run my credit nor do you have any permissible purpose to run my credit.   Since it is against the FCR A, 604. Permissible purposes of consumer reports [ 15 U.S.C. [[1681b]] ] for an entity  to view a consumers credit report without a permissible purpose. I am writing to inquire as to your alleged purpose for doing so since I did not apply for any credit with your company.   This inquiry was performed under false pretenses as described in the clear language of the la w. 15 USC 1681n ( a ) ( 1 ) ( B ) wh ich states, in part, in the case of liability of a natural person for obtaining a consumer report under false pretenses or knowingly without a permissible purpose, actual damages sustained by the consumer as a result of the failure or {$1000.00}, whichever is greater ;
    
    
    
    
    While checking my personal credit report, I discovered an Unauthorized and Fraudulent credit inquiry made without my KNOWLEDGE or CONSENT by XXXX on or about XXXX/XXXX/2014 on TRANSUNION credit file. I did not authorized or give permission to anyone employed by this company to make any inquiry and view my credit report. XXXX XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. I am requesting that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus and have them remove the unauthorized and fraudulent hard inquiry immediately.
    
    
    
    
    While checking my personal credit report, I noticed an unauthorized and fraudulent credit inquiry made by XXXX on or about XX/XX/XXXX onTransunion. I did not authorized anyone employed by this company to make any inquiry and view my credit report. XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. 
    I have requested that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus immediately and have them remove the unauthorized and fraudulent hard inquiry immediately. I also requested that they remove my personal information from their records. My Social Security # is XXXX and my Date of Birth is XX/XX/XXXX in case it is needed to locate the fraudulent inquiry in their system.
    
    
    
    
    While checking my personal credit report, I noticed an unauthorized and fraudulent credit inquiry made by XXXX on or about XX/XX/XXXX on Transunion. I did not authorized anyone employed by this company to make any inquiry and view my credit report. XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. 
    I have requested that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus immediately and have them remove the unauthorized and fraudulent hard inquiry immediately. I also requested that they remove my personal information from their records. My Social Security # is XXXX and my Date of Birth is XX/XX/XXXX in case it is needed to locate the fraudulent inquiry in their system.
    
    
    
    
    While checking my personal credit report, I noticed an unauthorized and fraudulent credit inquiry made by XXXX on or about XX/XX/XXXX on Transunion. I did not authorized anyone employed by this company to make any inquiry and view my credit report.XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. 
    I have requested that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus immediately and have them remove the unauthorized and fraudulent hard inquiry immediately. I also requested that they remove my personal information from their records. My Social Security # is XXXX and my Date of Birth is XX/XX/XXXX in case it is needed to locate the fraudulent inquiry in their system.
    
    
    
    
    While checking my personal credit report, I noticed an unauthorized and fraudulent credit inquiry made by XXXX on or about XX/XX/XXXX on Transunion. I did not authorized anyone employed by this company to make any inquiry and view my credit report. XXXX has violated the Fair Credit Reporting Act Section [[1681b]] ( c ). They were not legally entitled to make this fraudulent inquiry. This is a serious breach of my privacy rights. 
    I have requested that they mail me a copy of my signed authorization form that gave them the right to view my credit within five ( 5 ) business days so that I can verify its validity and advised them that if they can not provide me with proof that I authorized them to view my credit report then I am demanding that they contact the credit bureaus immediately and have them remove the unauthorized and fraudulent hard inquiry immediately. I also requested that they remove my personal information from their records. My Social Security # is XXXX and my Date of Birth is XXXX in case it is needed to locate the fraudulent inquiry in their system.
    
    
    
    


<a id='fin'></a>

# Conclusions

- The Logistic Regression model with metadata features confirms our observations that the products most indicative of receiving relief are credit cards and prepaid cards, bank and checking accounts.
- The model confirms that larger companies are more likely to award relief.
- Inspection of complaints text using the model coefficients, reveals some specific issues:
    - The Dodds-Frank law, codes 1681b and 1681g from the US consumer laws 
    - Length of complaint may be indicative of likelihood of relief (examples)
    - Company names are often embedded in complaints, so there is a possibility of overfitting to individual companies
    
#### Appendices

[Parameter Optimisation](#opt)
Code to optimise the logistic regression model using a randomised search.

[Thresholding predictions](#thresh)
The CFPB can control the level of recall and precision in the model predictions by applying a threshold against the prediction probabilities. In this way, they can understand the trade-off between the reduction in number of complaints they want to achieve using the model, and the proportion of potentially successful complaints they might miss by doing so.

[Latent Dirichlet Allocation](#lda)
Topic modelling is a helpful way to explore text data, and is an unsupervised method. pyLDAvis is used to visualise the output of an LDA model.

<a id='opt'></a>

# Parameter optimisation to get a strong model


```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfTransformer

def xx_remove(X):
    ## function to remove the xx blanks in strings, as this is noise
    fixed = X.str.replace('xx+','')
    return fixed



categorical_features = ['product', 'issue', 'sub-issue', 'company_size']
text_features = 'consumer_complaint_narrative'

categorical_transformer = Pipeline(steps=[
    ('ohe', OneHotEncoder(handle_unknown='ignore')),
    ('scaler', StandardScaler(with_mean=False))
])

text_transformer = Pipeline([
    ('xx', FunctionTransformer(xx_remove)),
    ('vect', CountVectorizer(strip_accents = 'unicode',
                                stop_words = 'english',
                                lowercase = True)),
    ('tfidf', TfidfTransformer())
])



preprocessor = ColumnTransformer(
transformers=[
    ('cat', categorical_transformer, categorical_features),
    ('text', text_transformer, text_features)
])

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```


```python
parameters = {
    'preprocessor__text__vect__max_df': (0.5, 0.75, 1.0),
    'preprocessor__text__vect__max_features': (None, 5000, 10000, 50000),
    'preprocessor__text__vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'preprocessor__text__tfidf__use_idf': (True, False),
    'preprocessor__text__tfidf__norm': ('l1', 'l2'),
    'preprocessor__cat__scaler':['passthrough', StandardScaler(with_mean=False)],
    'classifier__C':stats.uniform(0.1,10),
    'classifier__class_weight':[{1:w,0:1} for w in range(2,10)]
}

random_search = RandomizedSearchCV(clf, parameters, cv=3, n_iter=10, random_state=31415)
```


```python
random_search.fit(X_train, y_train, scoring=['f1_score','recall','precision'])
```


```python

preds = random_search.predict(X_test)
print(classification_report(y_test, preds))
```

                  precision    recall  f1-score   support
    
               0       0.88      0.82      0.85     10647
               1       0.45      0.57      0.51      2715
    
        accuracy                           0.77     13362
       macro avg       0.67      0.70      0.68     13362
    weighted avg       0.80      0.77      0.78     13362
    


#### Oversampling



```python
from imblearn.over_sampling import SMOTE
```


```python
smote = SMOTE(n_jobs=4)
X_train_smote, y_train_smote = smote.fit_sample(preprocessing.fit_transform(X_train), y_train)
```


```python
model = LogisticRegression(max_iter=200)
model.fit(X_train_smote, y_train_smote)
preds = model.predict(preprocessing.transform(X_test))
print(classification_report(y_test, preds))
```

<a id='thresh'></a>

# Prediction threshold


```python
preds = randomised_search.predict_proba(X_test)
for i in np.arange(0.3, 1, 0.1):
    print(i,'\n',classification_report(y_test, preds[:,1] > i))
```

<a id="clf"></a>

<a id='lda'></a>

# LDA


```python

vect = CountVectorizer(strip_accents = 'unicode',
                                stop_words = 'english',
                                lowercase = True,
                                max_df = 0.5, 
                                min_df = 100)

raw_text = df.dropna(subset=['consumer_complaint_narrative'])
raw_text.consumer_complaint_narrative = raw_text.consumer_complaint_narrative.apply(lambda x:x.lower())

lda = LatentDirichletAllocation(n_components=5) # one component for each product

encoded = vect.fit_transform(raw_text.consumer_complaint_narrative[raw_text.relief_received == 1].sample(10000))
lda.fit(encoded)
```

    C:\Users\thomas.merritt-smith\AppData\Local\Continuum\anaconda3\envs\uscfpb_test\lib\site-packages\pandas\core\generic.py:5208: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self[name] = value



```python
from pyLDAvis.sklearn import prepare
import pyLDAvis

vis = prepare(lda, encoded, vect,
#               mds='tsne'
             )

pyLDAvis.enable_notebook()

pyLDAvis.display(vis)


```


```python
topics = lda.transform(encoded)
text_with_topics = raw_text[raw_text.relief_received == 1]
text_with_topics['top_topic'] = topics.argmax(axis=1)
text_with_topics['top_topic_prob'] = topics.max(axis=1)

```


```python
text_with_topics.top_topic.value_counts()
```


```python
pd.crosstab(text_with_topics['issue'], text_with_topics.top_topic).sort_values(by=0, ascending=False)
```


```python
n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
print_top_words(lda, vect.get_feature_names(), n_top_words)

```
