
_Post script: This analysis was originally created in 2017, some 6 months before I actually got a job as a data scientist. This post is part of clearing a backlog of projects that I worked on before getting my first break as a Data Scientist. My approach is to publish as soon as it's not embarrassing - but there's still plenty of room for improvement_

## Getting strategic in the Statewide Nuisance Alligator Program (SNAP)




### Preamble

Florida Fish and Wildlife Conservation Commission's Statewide Alligator Harvest data between 2000 and 2016 is available at: http://myfwc.com/wildlifehabitats/managed/alligator/harvest/data-export

I was first made aware of the dataset by Jeremy Singer-Vine, through his data is plural newsletter of curious and interesting datasets: https://tinyletter.com/data-is-plural

Extra shoutout to LaTosha from FWC for your help in trying to link the data with some GISdata. I'd love to map some of this data out, but will leave that to the interested reader


#### Introduction

The analysis is set out in three parts:

* Clean the data (including converting the Carcass Size measurements to metric - sorry 'bout that US readers!)
* Explore the time-related fields to identify 'a good time to go hunting'
* Explore the location-related fields to identify 'a good place to go hunting'

_(Step 4. would be to fly out to Florida and test out my theories... #sponsorme!)_



```python
import seaborn as sns
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import re
pylab.rcParams['figure.figsize'] = 30, 20
%matplotlib inline


```


```python
raw_data = pd.read_csv('FWCAlligatorHarvestData.csv')
clean_data = deepcopy(raw_data)
```


```python
raw_data.loc[1:20]
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
      <th>Year</th>
      <th>Area Number</th>
      <th>Area Name</th>
      <th>Carcass Size</th>
      <th>Harvest Date</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>9 ft. 0 in.</td>
      <td>10-02-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>8 ft. 10 in.</td>
      <td>10-06-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>8 ft. 0 in.</td>
      <td>09-25-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>8 ft. 0 in.</td>
      <td>10-07-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>8 ft. 0 in.</td>
      <td>09-22-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>7 ft. 2 in.</td>
      <td>09-21-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>7 ft. 1 in.</td>
      <td>09-21-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>6 ft. 11 in.</td>
      <td>09-25-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>6 ft. 7 in.</td>
      <td>09-25-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>6 ft. 6 in.</td>
      <td>09-15-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>6 ft. 3 in.</td>
      <td>10-07-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2000</td>
      <td>102</td>
      <td>LAKE MARIAN</td>
      <td>12 ft. 7 in.</td>
      <td>09-04-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2000</td>
      <td>102</td>
      <td>LAKE MARIAN</td>
      <td>12 ft. 3 in.</td>
      <td>09-10-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2000</td>
      <td>102</td>
      <td>LAKE MARIAN</td>
      <td>12 ft. 3 in.</td>
      <td>09-03-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2000</td>
      <td>102</td>
      <td>LAKE MARIAN</td>
      <td>12 ft. 2 in.</td>
      <td>09-25-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2000</td>
      <td>102</td>
      <td>LAKE MARIAN</td>
      <td>12 ft. 0 in.</td>
      <td>09-27-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2000</td>
      <td>102</td>
      <td>LAKE MARIAN</td>
      <td>11 ft. 10 in.</td>
      <td>09-09-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2000</td>
      <td>102</td>
      <td>LAKE MARIAN</td>
      <td>11 ft. 7 in.</td>
      <td>09-10-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2000</td>
      <td>102</td>
      <td>LAKE MARIAN</td>
      <td>11 ft. 1 in.</td>
      <td>10-07-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2000</td>
      <td>102</td>
      <td>LAKE MARIAN</td>
      <td>11 ft. 1 in.</td>
      <td>10-07-2000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
raw_data.shape
```




    (87182, 6)



## Cleaning and preprocessing

Nothing revolutionary here:
- Convert lengths from imperial strings to metric floats
- look into null data and fill it where possible, or drop it


```python
def metric_size_converter(size):
    ## size is a string, with the measurements separated by spaces
    ## split this string to pull out the actual measurements
    
    string_list = size.split()
    feet = float(string_list[0])
    inches = float(string_list[2])

    ## convert feet and inches into metres
    metres = (feet*12 + inches)*2.54/100
    
    return metres
```


```python
clean_data['Carcass Size'] = clean_data['Carcass Size'].apply(lambda x: metric_size_converter(x))


```


```python
clean_data.head()


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
      <th>Year</th>
      <th>Area Number</th>
      <th>Area Name</th>
      <th>Carcass Size</th>
      <th>Harvest Date</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>3.4798</td>
      <td>09-22-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>2.7432</td>
      <td>10-02-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>2.6924</td>
      <td>10-06-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>2.4384</td>
      <td>09-25-2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>2.4384</td>
      <td>10-07-2000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
clean_data['Harvest_Date'] = pd.to_datetime(clean_data['Harvest Date'])


```


```python
clean_data.head()


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
      <th>Year</th>
      <th>Area Number</th>
      <th>Area Name</th>
      <th>Carcass Size</th>
      <th>Harvest Date</th>
      <th>Location</th>
      <th>Harvest_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>3.4798</td>
      <td>09-22-2000</td>
      <td>NaN</td>
      <td>2000-09-22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>2.7432</td>
      <td>10-02-2000</td>
      <td>NaN</td>
      <td>2000-10-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>2.6924</td>
      <td>10-06-2000</td>
      <td>NaN</td>
      <td>2000-10-06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>2.4384</td>
      <td>09-25-2000</td>
      <td>NaN</td>
      <td>2000-09-25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000</td>
      <td>101</td>
      <td>LAKE PIERCE</td>
      <td>2.4384</td>
      <td>10-07-2000</td>
      <td>NaN</td>
      <td>2000-10-07</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check for any null dates

len(clean_data[pd.isnull(clean_data['Harvest_Date'])])


```




    264




```python
clean_data[pd.isnull(clean_data['Harvest_Date'])].shape

## looks like there's no date for some catches, but there is a year


```




    (264, 12)




```python
# Still useful to have the year, but create a filter to avoid any Nan issues

has_date = pd.notnull(clean_data['Harvest_Date'])
has_date.shape

```




    (87145,)




```python
## Let's check that the year has still been recorded correctly

clean_data['Year'] = clean_data[has_date]['Harvest_Date'].apply(lambda x: x.year)

bad_date = clean_data[has_date][clean_data[has_date]['Year'] != clean_data[has_date]['Year']]

len(bad_date)


```




    0



_Always worth checking for internal consistency..._


```python
clean_data = clean_data.drop(bad_date.index, axis = 0)
clean_data.loc[~has_date,'Year'] = clean_data.loc[~has_date,'Year']
clean_data.shape
```




    (87182, 7)




```python
clean_data.loc[has_date,'Month'] = clean_data.loc[has_date,'Harvest_Date'].apply(lambda x: x.month)
clean_data.loc[has_date,'Day'] = clean_data.loc[has_date,'Harvest_Date'].apply(lambda x: x.day)
```


```python
len(clean_data.loc[~has_date,'Month'])


```




    264



# When should I go hunting?


```python
## How many distinct weeks do we have data for?

weeks = pd.DataFrame({'Year': clean_data.Year,'week':clean_data.Harvest_Date.apply(lambda x:x.week)})

weeks = weeks.drop_duplicates()

len(weeks)


```




    195




```python
import calendar as cal

days = list(cal.day_abbr)

clean_data['DayofWeek'] = clean_data['Harvest_Date'].apply(lambda x: x.dayofweek)



clean_data.loc[has_date,'Dayname'] = clean_data.loc[has_date,'DayofWeek'].apply(lambda x: days[int(x)])

day_count = clean_data.groupby('Dayname')['Carcass Size'].count()/(204)

day_count.reindex(days).plot(kind='bar').set(ylabel = 'Number of Carcasses', title = 'Average Carcasses per day 2000 - 2015')


```




    [Text(0, 0.5, 'Number of Carcasses'),
     Text(0.5, 1.0, 'Average Carcasses per day 2000 - 2015')]




![png](snap_analysis_files/snap_analysis_25_1.png)



```python
## Time series of number of catches over years

date_indexed_df = clean_data.set_index(pd.DatetimeIndex(clean_data['Harvest_Date']))

df_by_year = date_indexed_df.groupby(pd.TimeGrouper(freq='A'))

yearplot = df_by_year['Carcass Size'].count().plot(kind='line')

yearplot.set(xlabel='Year',ylabel='Total Carcasses')


```

    /home/tms/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:5: FutureWarning: pd.TimeGrouper is deprecated and will be removed; Please use pd.Grouper(freq=...)
      """





    [Text(0, 0.5, 'Total Carcasses'), Text(0.5, 0, 'Year')]




![png](snap_analysis_files/snap_analysis_26_2.png)



```python
fig, axarray = plt.subplots(4,4,figsize = (12,12), sharey = True)

row = col = 0
for key, grp in df_by_year:
    if col == 4:
        row += 1
        col = 0
    
    grp.groupby(pd.Grouper(freq='D'))['Carcass Size'].count().plot(ax = axarray[row,col],kind='line')
    col+=1
    
plt.tight_layout(pad=3)    
plt.suptitle('Daily Number of Carcasses by Year', size = 12)
plt.show()


```


![png](snap_analysis_files/snap_analysis_27_0.png)


### Observations

* There is a predictable weekday/weekend cycle
* There is usually a peak in the harvest towards the end of the season, which in recent years has been the end of November _(although in 2014 and 2015 catches took place outside of the September-November window)_
* The total harvest has gone up dramatically year on year

So, without a time machine to take us back to the quiet times around 2000-2005, the best time looks like a Monday towards the end of September.


# Where should I go hunting?


```python
## Each entry is marked with one or both of an Area Name and a Location e.g.

clean_data.iloc[87132]


```




    Year                           2015
    Area Number                     866
    Area Name             WALTON COUNTY
    Carcass Size                 2.6924
    Harvest Date             08-23-2015
    Location                 KINGS LAKE
    Harvest_Date    2015-08-23 00:00:00
    Month                             8
    Day                              23
    DayofWeek                         6
    Dayname                         Sun
    Name: 87132, dtype: object




```python
print('There are %i distinct areas, and %i distinct locations' %(len(clean_data['Area Name'].unique()),len(clean_data['Location'].unique())))


```

    There are 166 distinct areas, and 3577 distinct locations



```python
## How many entries don't have an Area Name? 101

Area_Unknown = clean_data['Area Name'].isnull()

print(Area_Unknown.sum())

## How many entries don't have a location? 71096

Location_Unknown = clean_data['Location'].isnull()

print(Location_Unknown.sum())

## How many don't have an area or a location? 99

Location_Unknown[Area_Unknown].sum()

## Can we infer the area from the location (when there's a location but no area?)

clean_data[Area_Unknown][~Location_Unknown]


```

    101
    71145


    /home/tms/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:19: UserWarning: Boolean Series key will be reindexed to match DataFrame index.





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
      <th>Year</th>
      <th>Area Number</th>
      <th>Area Name</th>
      <th>Carcass Size</th>
      <th>Harvest Date</th>
      <th>Location</th>
      <th>Harvest_Date</th>
      <th>Month</th>
      <th>Day</th>
      <th>DayofWeek</th>
      <th>Dayname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17244</th>
      <td>2006.0</td>
      <td>406</td>
      <td>NaN</td>
      <td>2.8448</td>
      <td>10-28-2006</td>
      <td>STA-5 (406)</td>
      <td>2006-10-28</td>
      <td>10.0</td>
      <td>28.0</td>
      <td>5.0</td>
      <td>Sat</td>
    </tr>
    <tr>
      <th>17272</th>
      <td>2006.0</td>
      <td>406</td>
      <td>NaN</td>
      <td>2.4130</td>
      <td>09-08-2006</td>
      <td>STA 5 (406)</td>
      <td>2006-09-08</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>Fri</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Looks like they're all from Area 406, which is sometimes called STA-5, 
## so let's just impute that and then drop any other unknown locations

clean_data.loc[clean_data['Area Number'] == 406,'Area Name'] = 'STA-5'


```


```python
Area_Unknown = clean_data['Area Name'].isnull()

clean_data = clean_data.drop(clean_data[Area_Unknown].index)


```

#### Sub-Areas

Some of our areas have a parentheses-enclosed sub-area, which may be useful to teases out the best place. If we just say _'Take me to St. John's River'_, it's better if we're referring to somewhere specific along that 300 mile stretch...!



```python
## Here are the specific sub areas

def sub_area_search(string):
    return re.search("\((.*?)\)",string,re.I)

sub_area_col = clean_data['Area Name'].apply(lambda x: sub_area_search(x).group(1) if sub_area_search(x) else None)

sub_area_filt = clean_data['Area Name'].apply(lambda x: True if sub_area_search(x) else False)

sub_area_col[sub_area_filt].unique()


```




    array(['POOL A', 'POOL C', 'WEST', 'SOUTH', 'NORTH', 'EAST',
           "LAKE HELL N' BLAZES", 'LAKE POINSETT', 'PUZZLE LAKE', 'WELAKA',
           'PALATKA SOUTH', 'WCAs 2A & 2B', 'WCAs 3A & 3B', 'BROADMOOR UNIT',
           'GOODWIN UNIT', 'WCA 2', 'WCA 3', 'PALATKA', 'POOL E'],
          dtype=object)




```python
clean_data[sub_area_filt].groupby('Area Name').Location.agg({'Distinct Locations' :  pd.Series.nunique,
                    'Total Harvest' : pd.Series.count})


```

    /home/tms/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: FutureWarning: using a dict on a Series for aggregation
    is deprecated and will be removed in a future version
      





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
      <th>Distinct Locations</th>
      <th>Total Harvest</th>
    </tr>
    <tr>
      <th>Area Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>EVERGLADES &amp; FRANCIS S. TAYLOR WMA (WCA 2)</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>EVERGLADES &amp; FRANCIS S. TAYLOR WMA (WCA 3)</th>
      <td>7</td>
      <td>9</td>
    </tr>
    <tr>
      <th>EVERGLADES &amp; FRANCIS S. TAYLOR WMA (WCAs 2A &amp; 2B)</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>EVERGLADES &amp; FRANCIS S. TAYLOR WMA (WCAs 3A &amp; 3B)</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>EVERGLADES WMA (WCA 2)</th>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>EVERGLADES WMA (WCA 3)</th>
      <td>26</td>
      <td>42</td>
    </tr>
    <tr>
      <th>KISSIMMEE RIVER (POOL A)</th>
      <td>13</td>
      <td>26</td>
    </tr>
    <tr>
      <th>KISSIMMEE RIVER (POOL C)</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>KISSIMMEE RIVER (POOL E)</th>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>LAKE OKEECHOBEE (EAST)</th>
      <td>24</td>
      <td>48</td>
    </tr>
    <tr>
      <th>LAKE OKEECHOBEE (NORTH)</th>
      <td>40</td>
      <td>80</td>
    </tr>
    <tr>
      <th>LAKE OKEECHOBEE (SOUTH)</th>
      <td>80</td>
      <td>231</td>
    </tr>
    <tr>
      <th>LAKE OKEECHOBEE (WEST)</th>
      <td>84</td>
      <td>174</td>
    </tr>
    <tr>
      <th>ST. JOHNS RIVER (LAKE HELL N' BLAZES)</th>
      <td>44</td>
      <td>92</td>
    </tr>
    <tr>
      <th>ST. JOHNS RIVER (LAKE POINSETT)</th>
      <td>67</td>
      <td>314</td>
    </tr>
    <tr>
      <th>ST. JOHNS RIVER (PALATKA SOUTH)</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ST. JOHNS RIVER (PALATKA)</th>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>ST. JOHNS RIVER (PUZZLE LAKE)</th>
      <td>24</td>
      <td>102</td>
    </tr>
    <tr>
      <th>ST. JOHNS RIVER (WELAKA)</th>
      <td>20</td>
      <td>43</td>
    </tr>
    <tr>
      <th>T.M. GOODWIN WMA (BROADMOOR UNIT)</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>T.M. GOODWIN WMA (GOODWIN UNIT)</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




<p>Let's take a look at the range of locations for a couple of these - should we just replace the area with the (sub area)?</p>



```python
clean_data[sub_area_filt][clean_data['Area Name'] == 'ST. JOHNS RIVER (WELAKA)'].groupby('Location').Year.count()


```

    /home/tms/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      """Entry point for launching an IPython kernel.





    Location
    ACCROSS FROM WEL                  1
    FL                                2
    LITTLE LAKE GEORGE                2
    NE SIDE OF LITTLE LAKE GEORGE     2
    PRIVATE PROPERTY                  4
    SEVEN SISTERS IS                  1
    SPORTSMANS HARBOR AREA            1
    ST JOHNS                          1
    ST JOHNS RIVER                    2
    ST JOHNS RIVER SHELL HABOR        1
    ST JOHNS RIVER-WALAKA             1
    ST JOHNS WELKA                    2
    ST. JOHNS RIVER                   1
    St. Johns River                   1
    TURKEY ISLAND COVE                1
    WALAKA                            1
    WELAKA                           14
    WELATKA 506                       2
    WELEKA                            1
    WELKA                             2
    Name: Year, dtype: int64




<p>So we'll just call it WELAKA as location - too much detail since they'll all have the same area code. Ultimately go back to the area code.</p>
<p>Also, looks like in most cases the location is unknown anyway</p>



```python
clean_data[clean_data['Area Name'] == 'ST. JOHNS RIVER (LAKE POINSETT)' ].groupby('Location').Year.count()


```




    Location
    502                                 6
    502 WASHINGTON                      1
    BREVARD                             1
    BREVARD COUNTY                      1
    Brevard                             1
    FL                                  1
    LAKE  POINCETT                      1
    LAKE  POINTSETTE                    1
    LAKE PAINSETTE                      1
    LAKE POINSET                        6
    LAKE POINSETE                       1
    LAKE POINSETT                      56
    LAKE POINSETT NORTH                 1
    LAKE POINSETTE                     15
    LAKE POINTSETT                      6
    LAKE PUINSATTA                      3
    LAKE WASH./ POINSETT                1
    LAKE WASHING 502                    1
    LAKE WASHINGTON                    72
    LAKE WASHINGTON 502                 1
    LAKE WASHINGTON DAM                 1
    LAKE WASHINGTON/POINSETT            1
    LAKE WINDER                        18
    LAKE WINER                          1
    LAKEPOINSETT                        1
    LK. WASHINGTON                      1
    Lake Harris                         1
    Lake Poinsett                       6
    Lake Poinsette                      3
    Lake poinsett                       1
                                       ..
    POINSETTA                           2
    POINSETTE                           4
    POINTSET                            2
    POINTSETT                           4
    PONSETT                             1
    SAINT JOHNS RIVER                   1
    SOUTH END OF LAKE WASHINGTON        2
    ST JOHNS                            2
    ST JOHNS IN BREVARD (               1
    ST JOHNS NORTH RIVER                1
    ST JOHNS RIVER                     13
    ST JOHNS RIVER LAKE POINSETT        1
    ST JOHNS RIVER LAKE POINSETTE       3
    ST JOHNS RIVER LAKE WINDER          2
    ST. JOHN IN BREVARD                 1
    ST. JOHNS RIVER                     7
    ST. JOHNS RIVER (LAKE POINSETT)     1
    ST.JOHNS                            2
    UPPER ST. JOHN                      1
    UPPER ST. JOHNS                     2
    WASHINGTN                           1
    WASHINGTON                          5
    WASHINGTON 502                      4
    WASHINGTTON                         1
    WASHINTON                           1
    WINDER                              2
    WINDSER                             1
    lake Poinsette                      1
    lake poinset                        2
    lake poinsett                       1
    Name: Year, Length: 67, dtype: int64




<p>Again, loads of locations, but the main two are Lake Poinsett and Lake Washington.</p>
<p>What's strange is that these lakes aren't particularly near each other, so maybe the location is not actually that helpful.</p>
<p>What we'll do is use the location to help indicate if the carcass was from a lake, river etc., but when looking for specific places, we'll only use the area</p>



```python

## simple regex function to generate new area columns

def area_classifier(area_name, specified_area_type):
    try:
        if re.search(specified_area_type,area_name, re.I):
            return True
        else:
            return False
    except:
        return False


```


```python
filtarea_names = ['River','Lake','County','WMA','Marsh','Pond', 'Unknown','Reservoir','^STA']


```


```python
filt_area_dict = {}

for n in filtarea_names:
    filt_area_dict[n]=clean_data['Area Name'].apply(lambda x: area_classifier(x, n)) 


## Make these boolean filters rather than columns on the dataframe
## Start with locations where available then go on to areas
## Little River Lake is a Lake, not a river.


```


```python
clean_data[filt_area_dict['WMA']]['Area Name'].unique()
```




    array(['EVERGLADES WMA - 2', 'EVERGLADES WMA - 3', 'HOLEY LAND WMA',
           'GUANA RIVER WMA', 'OCALA WMA', 'THREE LAKES WMA',
           'EVERGLADES & FRANCIS S. TAYLOR WMA (WCAs 2A & 2B)',
           'EVERGLADES & FRANCIS S. TAYLOR WMA (WCAs 3A & 3B)',
           'T.M. GOODWIN WMA (BROADMOOR UNIT)',
           'T.M. GOODWIN WMA (GOODWIN UNIT)', 'EVERGLADES WMA (WCA 2)',
           'EVERGLADES WMA (WCA 3)', 'T.M. GOODWIN WMA GOODWIN UNIT)',
           'EVERGLADES & FRANCIS S. TAYLOR WMA (WCA 2)',
           'EVERGLADES & FRANCIS S. TAYLOR WMA (WCA 3)'], dtype=object)




```python
filt_location_dict = {}
filtlocation_names = ['River','Lake','County','WMA','Marsh','Pond', 'Unknown','Swamp','Creek','Canal','Springs']

for n in filtlocation_names:
    
    filt_location_dict[n] = clean_data['Location'].apply(lambda x: area_classifier(x, n))


```


```python
clean_data[filt_location_dict['Swamp']][filt_area_dict['Lake']]
```

    /home/tms/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      """Entry point for launching an IPython kernel.





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
      <th>Year</th>
      <th>Area Number</th>
      <th>Area Name</th>
      <th>Carcass Size</th>
      <th>Harvest Date</th>
      <th>Location</th>
      <th>Harvest_Date</th>
      <th>Month</th>
      <th>Day</th>
      <th>DayofWeek</th>
      <th>Dayname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>78559</th>
      <td>2014.0</td>
      <td>835</td>
      <td>LAKE COUNTY</td>
      <td>3.5560</td>
      <td>09-27-2014</td>
      <td>OKAHUMPKA SWAMP</td>
      <td>2014-09-27</td>
      <td>9.0</td>
      <td>27.0</td>
      <td>5.0</td>
      <td>Sat</td>
    </tr>
    <tr>
      <th>78675</th>
      <td>2014.0</td>
      <td>835</td>
      <td>LAKE COUNTY</td>
      <td>2.6924</td>
      <td>09-27-2014</td>
      <td>OKAHUMPKA SWAMP</td>
      <td>2014-09-27</td>
      <td>9.0</td>
      <td>27.0</td>
      <td>5.0</td>
      <td>Sat</td>
    </tr>
    <tr>
      <th>85658</th>
      <td>2015.0</td>
      <td>835</td>
      <td>LAKE COUNTY</td>
      <td>2.1336</td>
      <td>10-17-2015</td>
      <td>GREEN SWAMP</td>
      <td>2015-10-17</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>5.0</td>
      <td>Sat</td>
    </tr>
    <tr>
      <th>85703</th>
      <td>2015.0</td>
      <td>835</td>
      <td>LAKE COUNTY</td>
      <td>1.7018</td>
      <td>10-11-2015</td>
      <td>GREEN SWAMP</td>
      <td>2015-10-11</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>Sun</td>
    </tr>
  </tbody>
</table>
</div>




```python
{i:filt_location_dict[i].sum() for i in filt_location_dict}
```




    {'River': 4282,
     'Lake': 6234,
     'County': 132,
     'WMA': 34,
     'Marsh': 46,
     'Pond': 304,
     'Unknown': 1,
     'Swamp': 149,
     'Creek': 1544,
     'Canal': 573,
     'Springs': 44}




```python
## Find anything which has a double count in location (e.g. Newmans lake will look like a 'wma' and a 'lake') 
from itertools import combinations as cmb

overlap_locations = {c:(filt_location_dict[c[0]] & filt_location_dict[c[1]]) 
                     for c in cmb(filt_location_dict.keys(),2)}

overlap_locations_counts = {l:overlap_locations[l].sum() for l in overlap_locations if overlap_locations[l].sum()!=0}

overlap_locations_counts


```




    {('River', 'Lake'): 19,
     ('River', 'County'): 9,
     ('River', 'Swamp'): 36,
     ('River', 'Creek'): 9,
     ('River', 'Canal'): 3,
     ('River', 'Springs'): 3,
     ('Lake', 'County'): 12,
     ('Lake', 'WMA'): 20,
     ('Lake', 'Marsh'): 12,
     ('Lake', 'Pond'): 7,
     ('Lake', 'Swamp'): 5,
     ('Lake', 'Creek'): 19,
     ('Lake', 'Canal'): 45,
     ('Lake', 'Springs'): 2,
     ('County', 'Pond'): 6,
     ('County', 'Creek'): 2,
     ('County', 'Canal'): 16,
     ('County', 'Springs'): 2,
     ('Pond', 'Swamp'): 1,
     ('Pond', 'Creek'): 12,
     ('Pond', 'Canal'): 19,
     ('Swamp', 'Creek'): 40,
     ('Creek', 'Canal'): 6,
     ('Creek', 'Springs'): 16}




```python
overlap_locationsdata = {l:clean_data.Location[overlap_locations[l]].unique() for l in overlap_locations if overlap_locations[l].sum()!=0}

overlap_locationsdata


```




    {('River',
      'Lake'): array(['ST JOHNS RIVER LAKE POINSETT', 'ST JOHNS RIVER LAKE WINDER',
            'ST. JOHNS RIVER (LAKE POINSETT)', 'LAKE HARNEY ST JOHNS RIVER',
            'LAKE HARNEY (ST. JOHNS RIVER)', 'ST HONS RIVER LAKE HARNEY',
            'KISS RIVER NORTH LAKE', 'SMOKEHOUSE LAKE/CHOCTACHATEE RIVER',
            'ST JOHNS RIVER/LAKE HELL N BLAZES',
            'ST JOHNS RIVER LAKE POINSETTE', 'ST JOHNS RIVER/LAKE HARNEY',
            'MANATEE RIVER/LAKE MANATEE EAST OF VERNA BETHANY R',
            'choctawhatchee river-- inside lake', 'LITTLE RIVER LAKE'],
           dtype=object),
     ('River',
      'County'): array(['CLAY COUNTY, ST. JOHNS RIVER', 'NORTH INDIAN RIVER COUNTY',
            'NASSAU COUNTY, ST MARYS RIVER', 'OKALOOSA COUNTY, YELLOW RIVER',
            'River/creek off of county rd 67',
            'Escambia River (santa rosa county side)',
            'escambia River (santa rosa county side)'], dtype=object),
     ('River', 'Swamp'): array(['SIMPSON RIVER, ESCAMBIA RIVER SWAMP',
            'WHITE RIVER, ESCAMBIA RIVER SWAMP', 'WITHLACOOCHEE RIVER SWAMP',
            'RIVER SWAMP'], dtype=object),
     ('River', 'Creek'): array(['St. Johns River Deep Creek',
            'JUST OFF APALACH. RIVER IN GRASSY CREEK',
            'SMOKEHOUSE CREEK-BROTHERS RIVER', 'ST. JOHNS RIVER, DEEP CREEK',
            'ST. JOHNS RIVER, TROUT CREEK', 'River/creek off of county rd 67',
            'St. Johns River north of deep creek',
            'Tocoi Creek off of St. Johns River'], dtype=object),
     ('River',
      'Canal'): array(['KISSIMMEE RIVER CANAL', 'ST JOHNS RIVER UPPER BASIN CANALS'],
           dtype=object),
     ('River',
      'Springs'): array(['Santa Fe River. High springs', 'LITTLE RIVER SPRINGS'],
           dtype=object),
     ('Lake', 'County'): array(['Lake County', 'lake county blue springs',
            'lake county blue springs area', 'Polk County Lake', 'LAKE COUNTY',
            'PRIVATE POND IN SOUTH LAKE COUNTY', 'PIVATE LAKE IN LEE COUNTY',
            'Lake county/ lake denham', 'Lake county / private land',
            'polk county private lake no name'], dtype=object),
     ('Lake', 'WMA'): array(['NEWMANS LAKE'], dtype=object),
     ('Lake',
      'Marsh'): array(['Lettuce Lake (Murphy Marsh)', 'LETTUCE LAKE (MURPHY MARSH)',
            'LETTUCE LAKE / MURPHY MARSH', 'MURPHY MARSH (LETTUCE LAKE)',
            'MURPHY MARSH/LETTUCE LAKE', 'LETTUCE LAKE(MURPHYMARSH)',
            'LAKE MATTIE MARSH'], dtype=object),
     ('Lake',
      'Pond'): array(['LEISURE LAKE POND', 'PRIVATE PROPERTY (POND) SR 70 LAKE PLACID',
            'PRIVATE POND IN SOUTH LAKE COUNTY', 'Private Pond Lakeland',
            'Private Pond LakeLand', 'POND ON MOORE RD LAKELAND',
            'BRACEY RD N LAKELAND POND'], dtype=object),
     ('Lake', 'Swamp'): array(['GAP LAKE SWAMP', 'SWAMP, LAKE'], dtype=object),
     ('Lake',
      'Creek'): array(['ECONFINA CREEK, DEERPOINT LAKE', 'LAKE MARION CREEK',
            'BLAKESLEE CREEK', 'LAKE BUTLER CREEK', 'LAKE ASHBY CREEK'],
           dtype=object),
     ('Lake', 'Canal'): array(['LAKE JUNE CANAL', 'INDAIN LAKE ESTATES CANAL',
            'RIM CANAL LAKE OKEECHOBEE', 'LAKE OKEECHOBEE RIM CANAL',
            'GASPRILLIA ROAD CANAL @HOLIDAY LAKES', 'LAKE GENTRY CANAL',
            'CANAL OFF LAKE CYPRESS', 'WEST LAKE CANAL',
            'WEST LAKE TOHO PRIVATE CANAL',
            'PRIVATE CANAL OFF OF WEST LAKE TOHO',
            'Canal adjacent to Lake Okeechobee', 'CANAL LAKE',
            'johns lake canal', 'Canal going from w lake toho to cypress lake',
            'canal that goes from w lake toho to cypress lake',
            'Lake cypress canals', 'Lake gentry canal', 'GANT LAKE CANAL',
            'LAKE ASHBY CANAL', 'LAKE TARPON CANAL'], dtype=object),
     ('Lake',
      'Springs'): array(['lake county blue springs', 'lake county blue springs area'],
           dtype=object),
     ('County', 'Pond'): array(['NE CORNER HOLMES COUNTY, BAXLEY POND',
            'PRIVATE POND IN SOUTH LAKE COUNTY',
            'private pond eastern Lafayette County',
            'Washington County / Private Property Pond HWY 77',
            'Back Waters of Gap Pond hwy 77 Washington County'], dtype=object),
     ('County', 'Creek'): array(['River/creek off of county rd 67',
            'creek of of county rd 67 in franklin county'], dtype=object),
     ('County',
      'Canal'): array(['Canals off of interceptor lagoon Charlotte county',
            'GULF COUNTY CANAL', 'COUNTY LINE CANAL'], dtype=object),
     ('County',
      'Springs'): array(['lake county blue springs', 'lake county blue springs area'],
           dtype=object),
     ('Pond', 'Swamp'): array(['Swamp Pond'], dtype=object),
     ('Pond', 'Creek'): array(['SWIFT CREEK POND'], dtype=object),
     ('Pond',
      'Canal'): array(['Harney Pond Canal', 'Harne Pond Canal', 'Hawney Pond Canal',
            'HARNEY POND CANAL', 'Near harney pond canal'], dtype=object),
     ('Swamp',
      'Creek'): array(['REEDY CREEK SWAMP', 'RICE CREEK SWAMP'], dtype=object),
     ('Creek', 'Canal'): array(['FIRST EATEN CREEK CANAL', 'TAYLOR CREEK CANAL',
            'CANOE CREEK CANAL'], dtype=object),
     ('Creek', 'Springs'): array(['ALEXANDER SPRINGS CREEK'], dtype=object)}




```python
## Find anything which has a double count in Area (e.g. Newmans lake)

overlap_area = {c:(filt_area_dict[c[0]] & filt_area_dict[c[1]]) 
                     for c in cmb(filt_area_dict.keys(),2)}

overlap_area_counts = {l:overlap_area[l].sum() for l in overlap_area if overlap_area[l].sum()!=0}

overlap_area_counts


```




    {('River', 'Lake'): 9763,
     ('River', 'County'): 39,
     ('River', 'WMA'): 153,
     ('Lake', 'County'): 2466,
     ('Lake', 'WMA'): 176}




```python
overlap_areadata = {l:clean_data['Area Name'][overlap_area[l]].unique() for l in overlap_area if overlap_area[l].sum()!=0}

overlap_areadata


```




    {('River', 'Lake'): array(["ST. JOHNS RIVER (LAKE HELL N' BLAZES)",
            'ST. JOHNS RIVER (LAKE POINSETT)', 'ST. JOHNS RIVER (PUZZLE LAKE)'],
           dtype=object),
     ('River', 'County'): array(['INDIAN RIVER COUNTY'], dtype=object),
     ('River', 'WMA'): array(['GUANA RIVER WMA'], dtype=object),
     ('Lake', 'County'): array(['LAKE COUNTY'], dtype=object),
     ('Lake', 'WMA'): array(['THREE LAKES WMA'], dtype=object)}



This counting thing is never going to be 100% correct, unless we went through and relabelled these by hand. We're not going to do that, so we'll make clear a few _sensible_ rules that we're following to do the counting:

#### 1. Prioritise the Location before the Area  

#### 2. Dealing with Double labelled **Locations**
    
* If it's a county, use the other label</li>
* If it's a lake and a river, call it a lake</li>
* If it's a lake and a WMA, it's a lake</li>
* If it's labelled as a creek, pond, canal, swamp or marsh and something else, ignore the other thing.  


#### 3. Dealing with double labelled **Areas**
* If it's a county, use the other label
* If it's a lake and a river, it's a lake
* If it's a WMA and something else, ignore the other thing  


#### 4. Where there are different labels across the Area and Location, take the _Location_  
    
#### 5. If you still can't classify it, it's just somewhere in the county. Hollows, pools, ditches or 2 miles down from Eureka, them gators get everywhere!  



<h3 id="This-counting-thing-is-never-going-to-be-100%-correct,-unless-we-went-through-and-relabelled-these-by-hand.">This counting thing is never going to be 100% correct, unless we went through and relabelled these by hand.<a class="anchor-link" href="#This-counting-thing-is-never-going-to-be-100%-correct,-unless-we-went-through-and-relabelled-these-by-hand.">¶</a></h3><p>We're not going to do that, so we'll make clear a few <em>sensible</em> rules that we're following to do the counting:</p>
<h4 id="1.-Prioritise-the-Location-before-the-Area">1. Prioritise the Location before the Area<a class="anchor-link" href="#1.-Prioritise-the-Location-before-the-Area">¶</a></h4><h4 id="2.-Dealing-with-Double-labelled-Locations">2. Dealing with Double labelled <em>Locations</em><a class="anchor-link" href="#2.-Dealing-with-Double-labelled-Locations">¶</a></h4><ul>
<li>If it's a county, use the other label</li>
<li>If it's a lake and a river, call it a lake</li>
<li>If it's a lake and a WMA, it's a lake</li>
<li>If it's labelled as a creek, pond, canal, swamp or marsh and something else, ignore the other thing. </li>
</ul>
<h4 id="3.-Dealing-with-double-labelled-Areas">3. Dealing with double labelled <em>Areas</em><a class="anchor-link" href="#3.-Dealing-with-double-labelled-Areas">¶</a></h4><ul>
<li>If it's a county, use the other label</li>
<li>If it's a lake and a river, it's a lake</li>
<li>If it's a WMA and something else, ignore the other thing</li>
</ul>
<h4 id="4.-Where-there-are-different-labels-across-the-Area-and-Location,-take-the-Location">4. Where there are different labels across the Area and Location, take the Location<a class="anchor-link" href="#4.-Where-there-are-different-labels-across-the-Area-and-Location,-take-the-Location">¶</a></h4><h4 id="5.-If-you-still-can't-classify-it,-it's-just-somewhere-in-the-county.-Hollows,-pools,-ditches-or-2-miles-down-from-Eureka,-them-gators-get-everywhere!">5. If you still can't classify it, it's just somewhere in the county. Hollows, pools, ditches or 2 miles down from Eureka, them gators get everywhere!<a class="anchor-link" href="#5.-If-you-still-can't-classify-it,-it's-just-somewhere-in-the-county.-Hollows,-pools,-ditches-or-2-miles-down-from-Eureka,-them-gators-get-everywhere!">¶</a></h4>



```python
conditions = [
    filt_location_dict['Creek'],
    filt_location_dict['Canal'],
    filt_location_dict['Springs'],
    filt_location_dict['Pond'],
    filt_location_dict['Swamp'],
    filt_location_dict['Marsh'],
    filt_location_dict['Lake'],
    filt_location_dict['River'],
    filt_location_dict['WMA'],
    filt_area_dict['WMA'],
    filt_area_dict['^STA'],
    filt_area_dict['Marsh'],
    filt_area_dict['Reservoir'],
    (filt_area_dict['Lake']) & (filt_area_dict['County']),
    (filt_area_dict['River']) & (filt_area_dict['County']),
    filt_area_dict['Lake'],
    filt_area_dict['River'],
    filt_area_dict['County']]

choices = ['Creek','Canal','Springs','Pond', 'Swamp', 'Marsh','Lake','River','WMA','WMA','STA','Marsh','Reservoir','County','County','Lake','River','County']

clean_data['geography'] = np.select(conditions, choices, default='Unknown')


```


```python
clean_data[clean_data['geography'] == 'Unknown']['Carcass Size'].count()


```




    463




```python
### So we've got about 0.5% carcasses with no geography

clean_data[clean_data['geography'] == 'Unknown'].groupby(['Area Number','Area Name'])['Carcass Size'].count().sort_values(ascending = False)


```




    Area Number  Area Name                             
    542          BLUE CYPRESS WATER MANAGEMENT AREA        340
    112          TENEROC FMA                                39
    401          A.R.M. LOXAHATCHEE NWR                     36
    546          T.M. GOODWIN WATERFOWL MANAGEMENT AREA     29
    112          TENOROC FMA                                15
    839          LIBERTY                                     2
    819          FRANKLIN                                    2
    Name: Carcass Size, dtype: int64




```python
clean_data.groupby('geography')['Carcass Size'].count().sort_values(ascending = False).plot(kind = 'bar', 
                                                                                            title = 'Carcass count by geography')


```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f38e83cfac8>




![png](snap_analysis_files/snap_analysis_59_1.png)



```python
clean_data[clean_data['geography'] == 'Lake'].groupby('Area Name')['Carcass Size'
                                            ].count().sort_values(ascending = False)[:25].plot(kind = 'bar', 
                                                                                    title = 'Carcass count by Lake Area')


```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f38e3600a58>




![png](snap_analysis_files/snap_analysis_60_1.png)



```python
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9),(ax10,ax11,ax12)) = plt.subplots(4,3,figsize = (8,25))
axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]

geography_names = list(clean_data['geography'].unique())

geography_names.remove('Unknown')

for i, area in enumerate(geography_names):
    try:
        d = clean_data[clean_data.geography == area].groupby(['Area Name'])['Carcass Size'
                                ].count().sort_values(ascending = False)
        if len(d) > 10:
            d[:10].plot(ax = axes[i],kind = 'bar', title = 'Top 10 %s' %area)

        else:
            d.plot(ax = axes[i],kind = 'bar', title = 'Top 10 %s' %area)
    except:
        pass
    
    
plt.tight_layout()
plt.show()


```


![png](snap_analysis_files/snap_analysis_61_0.png)


### Range of sizes


```python

#average size over time
#range of sizes vs geography
#range of sizes vs top locations


```


```python
## first off, identify any lines where the Carcass Size is 0

zero_size = clean_data[clean_data['Carcass Size']==0].index

## fill these missing data points with the average Carcass Size

clean_data.loc[zero_size,'Carcass Size'] = clean_data[clean_data['Carcass Size']!=0]['Carcass Size'].mean()


```


```python
clean_data['Carcass Size'].plot(kind='box')


```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f38e35bef28>




![png](snap_analysis_files/snap_analysis_65_1.png)



```python
## Identify the biggest catches ever

clean_data[clean_data['Carcass Size'] == clean_data['Carcass Size'].max()]


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
      <th>Year</th>
      <th>Area Number</th>
      <th>Area Name</th>
      <th>Carcass Size</th>
      <th>Harvest Date</th>
      <th>Location</th>
      <th>Harvest_Date</th>
      <th>Month</th>
      <th>Day</th>
      <th>DayofWeek</th>
      <th>Dayname</th>
      <th>geography</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44996</th>
      <td>2010.0</td>
      <td>502</td>
      <td>ST. JOHNS RIVER (LAKE POINSETT)</td>
      <td>4.3434</td>
      <td>10-31-2010</td>
      <td>NaN</td>
      <td>2010-10-31</td>
      <td>10.0</td>
      <td>31.0</td>
      <td>6.0</td>
      <td>Sun</td>
      <td>Lake</td>
    </tr>
    <tr>
      <th>78315</th>
      <td>2014.0</td>
      <td>828</td>
      <td>HIGHLANDS COUNTY</td>
      <td>4.3434</td>
      <td>10-28-2014</td>
      <td>LITTLE RED WATER LAKE</td>
      <td>2014-10-28</td>
      <td>10.0</td>
      <td>28.0</td>
      <td>1.0</td>
      <td>Tue</td>
      <td>Lake</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Which locations have the most 4m (14 feet) or greater alligators?

clean_data[clean_data['Carcass Size'] >= 4].groupby(['Area Name'])['Carcass Size'].count().sort_values(ascending = False)[:10]


```




    Area Name
    LAKE GEORGE               10
    PUTNAM COUNTY              8
    LAKE JESUP                 5
    LAKE TOHOPEKALIGA          5
    LAKE COUNTY                5
    HIGHLANDS COUNTY           5
    CRESCENT LAKE              4
    LAKE OKEECHOBEE (WEST)     4
    LAKE GRIFFIN               3
    KISSIMMEE RIVER            3
    Name: Carcass Size, dtype: int64




```python
clean_data['Carcass Size'].plot(kind='hist',bins = 50, title = 'Distribution of catches by carcass size')


```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f38e2849080>




![png](snap_analysis_files/snap_analysis_68_1.png)



```python
df_by_year['Carcass Size'].mean().plot(title = 'Carcass Size by month')


```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f38e27a6c50>




![png](snap_analysis_files/snap_analysis_69_1.png)



<p>Interesting to note the dip in size around 2010.</p>



```python
clean_data.groupby(['Area Name']).filter(lambda x: x['Carcass Size'].count() > 1000
                                        ).groupby(['Area Name'])['Carcass Size'].mean(
                                        ).sort_values(ascending = False).plot(kind='bar', title='Average Carcass Size in Areas with more than 1000 catches')


```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f38e2288400>




![png](snap_analysis_files/snap_analysis_71_1.png)



```python
plot = sns.violinplot(x = clean_data['geography'], y = clean_data['Carcass Size'])

plot.set_xticklabels(labels = geography_names,rotation = 90)

plt.show()
```


![png](snap_analysis_files/snap_analysis_72_0.png)



<p>What about if I want to look for some smaller Alligators to get my eye in first?</p>



```python
clean_data.groupby(['Area Name'])['Carcass Size'].agg({'average size': 'mean'}).sort_values(by='average size',ascending = True)[:10]


```

    /home/tms/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: FutureWarning: using a dict on a Series for aggregation
    is deprecated and will be removed in a future version
      """Entry point for launching an IPython kernel.





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
      <th>average size</th>
    </tr>
    <tr>
      <th>Area Name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BRADFORD COUNTY</th>
      <td>2.168309</td>
    </tr>
    <tr>
      <th>FRANKLIN</th>
      <td>2.171700</td>
    </tr>
    <tr>
      <th>BAKER COUNTY</th>
      <td>2.225239</td>
    </tr>
    <tr>
      <th>UNION COUNTY</th>
      <td>2.236519</td>
    </tr>
    <tr>
      <th>LAKE ROUSSEAU</th>
      <td>2.245573</td>
    </tr>
    <tr>
      <th>SUWANNEE COUNTY</th>
      <td>2.249251</td>
    </tr>
    <tr>
      <th>OKALOOSA COUNTY</th>
      <td>2.251529</td>
    </tr>
    <tr>
      <th>PEACE RIVER NORTH</th>
      <td>2.257245</td>
    </tr>
    <tr>
      <th>WASHINGTON COUNTY</th>
      <td>2.275885</td>
    </tr>
    <tr>
      <th>SANTA ROSA COUNTY</th>
      <td>2.288073</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(clean_data['Carcass Size'].min())

clean_data[clean_data['Carcass Size'] <= 1].groupby(['Area Name']
                            )['Carcass Size'].agg({'Number of catches less than 1m':'count'}
                            ).sort_values(by='Number of catches less than 1m',ascending = False)[:10]


```

    0.3048


    /home/tms/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:4: FutureWarning: using a dict on a Series for aggregation
    is deprecated and will be removed in a future version
      after removing the cwd from sys.path.





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
      <th>Number of catches less than 1m</th>
    </tr>
    <tr>
      <th>Area Name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ORANGE LAKE</th>
      <td>23</td>
    </tr>
    <tr>
      <th>KISSIMMEE RIVER</th>
      <td>11</td>
    </tr>
    <tr>
      <th>ST. JOHNS RIVER (PUZZLE LAKE)</th>
      <td>10</td>
    </tr>
    <tr>
      <th>LAKE OKEECHOBEE (SOUTH)</th>
      <td>10</td>
    </tr>
    <tr>
      <th>LAKE HARNEY</th>
      <td>9</td>
    </tr>
    <tr>
      <th>LOCHLOOSA LAKE</th>
      <td>9</td>
    </tr>
    <tr>
      <th>LAKE TOHOPEKALIGA</th>
      <td>7</td>
    </tr>
    <tr>
      <th>LAKE OKEECHOBEE (WEST)</th>
      <td>6</td>
    </tr>
    <tr>
      <th>LAKE HATCHINEHA</th>
      <td>6</td>
    </tr>
    <tr>
      <th>POLK COUNTY</th>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




<p>So, if you're looking for some big gators, get down to Lake Jesup (biggest average catch) or Lake George (most catches above 4m), and if you want something smaller to start off with, get to Bradford County (smallest average catch) or Orange Lake (largest number of catches less than 1m)</p>


### Awwww, SNAP... That's all we've got time for. Catch ya!



Tom Merritt Smith 2017
Get in touch: tmerrittsmith (at) gmail (dot) com


#### Postscript: March, 2019

A few things I would do next time:

- map some of this stuff
- consider something like vega/altair to make interactive plots
