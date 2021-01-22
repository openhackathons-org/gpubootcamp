#!/usr/bin/env python
# coding: utf-8

import cudf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from timeit import default_timer as timer
import pickle

import cuml
#import sklearn
#import sklearn.svm
#import sklearn.neighbors

def run_everything():
    # ### 2.1 Prepare weather data
    # First, we will download the weather data.

    # In[2]:

    t_start = timer()

    filename = 'data/weather2011-2012.csv'


    # cuDF DataFrames are a tabular structure of data that reside on the GPU. We interface with these cuDF DataFrames in the same way we interface with Pandas DataFrames that reside on the CPU - with a few deviations. Load data from CSV file into a cuDF DataFrame.

    # In[3]:

    t_file_start = timer()
    weather = cudf.read_csv(filename)
    t_file = timer() - t_file_start

    # #### 2.1.1 Inspecting a cuDF DataFrame
    #
    # There are several ways to inspect a cuDF DataFrame. The first method is to enter the cuDF DataFrame directly into the REPL. This shows us an overview about the DatFrame including its type and metadata such as the number of rows or columns.

    # In[4]:




    # A second way to inspect a cuDF DataFrame is to wrap the object in a Python print function `print(weather)` function. This results in showing the rows and columns of the dataframe with simple formating.
    #
    # For very large dataframes, we often want to see the first couple rows. We can use the `head` method of a cuDF DataFrame to view the first N rows.

    # In[5]:




    # #### 2.1.2 Columns
    #
    # cuDF DataFrames store metadata such as information about columns or data types. We can access the columns of a cuDF DataFrame using the `.columns` attribute.

    # In[6]:




    # We can modify the columns of a cuDF DataFrame by modifying the `columns` attribute. We can do this by setting that attribute equal to a list of strings representing the new columns. Let's shorten the two longest column names!

    # In[7]:


    ### TODO rename the relative temperature column to RTemp, and the relative humidity to Humidity
    #weather.columns = ['Hour', 'Temperature', 'Relative Temperature', 'Rel. Humidity', 'Wind', 'Weather']
    weather.columns = ['Hour', 'Temperature', 'RTemp', 'Humidity', 'Wind', 'Weather']


    # #### 2.1.3 Series
    #
    # cuDF DataFrames are composed of rows and columns. Each column is represented using an object of type `Series`. For example, if we subset a cuDF DataFrame using just one column we will be returned an object of type `cudf.dataframe.series.Series`.

    # In[8]:


    humidity = weather['Humidity']
    #print(type(humidity))
    #print(humidity)


    # We also see a column of values on the left hand side with values 0, 1, 2, 3. These values represent the index of the Series.
    # The DataFrame and Series objects have both an index attribute that will be useful for joining tables and also for selecting data.

    # #### 2.1.4 Data Types
    #
    # We can also inspect the data types of the columns of a cuDF DataFrame using the `dtypes` attribute.

    # In[9]:


    #print(weather.dtypes)


    # We can modify the data types of the columns of a cuDF DataFrame by passing in a cuDF Series with a modified data type.

    # In[10]:


    weather['Humidity'] = weather['Humidity'].astype(np.float64)
    #print(weather.dtypes)


    # The 'Weather' column provides a description of the weather condidions. We should mark it as a categorical column.

    # In[11]:


    weather['Weather'] = weather['Weather'].astype('category')


    # After this step the numerical category codes can be accessed using the `.cat.codes` attribute of the column. We actually will not need the category labels, we just replace the 'Weather' column with the category codes.

    # In[12]:


    weather['Weather'] = weather['Weather'].cat.codes


    # The data type of the 'Hour' column is `object` which means a string. Let's convert this to a numeric value! This cannot be done with the `astype` method, you should use the [cudf.to_datetime](https://docs.rapids.ai/api/cudf/nightly/api.html#cudf.to_datetime) function!

    # In[13]:


    ### TODO convert the 'Hour' column from string to datetime
    weather['Hour'] = cudf.to_datetime(weather['Hour'])


    # #### 2.1.2 Prepare features
    # ##### Operations with cudf Series
    # We can perform mathematical operations on the Series data type. We will scale the Humidity and and Temperature variables, so that they lay in the [0, 1] range (some ML algorithms work better if the input data is scaled this way).

    # In[14]:


    weather['Humidity'] = weather['Humidity'] / 100.0


    # We will scale the temperature using the following formula T = (T - Tmin) / (Tmax - Tmin). First we select the min and max values.

    # In[15]:


    T = weather['Temperature']

    # Select the minimum temperature
    Tmin = T.min()

    ### TODO select the maximum temperature (1 line of code)
    Tmax = T.max()

    #print(Tmin, Tmax)


    # We could simply use the Tmin and Tmax values and apply the above formula on the series.
    #
    # ##### User defined functions (UDF)
    # We can write custom functions to operate on the data. When cuDF executes a UDF, it gets just-in-time (JIT) compiled into a CUDA kernel (either explicitly or implicitly) and is run on the GPU. Let's write a function that scales the temperature!

    # In[16]:


    def scale_temp(T):
        # Note that the Tmin and Tmax variables are stored during compilation time and remain constant afterwards
        T = (T - Tmin) / (Tmax - Tmin)
        return T


    # The applymap function will call scale_temp on all element of the series

    # In[17]:


    weather['Temperature'] = weather['Temperature'].applymap(scale_temp)


    # Lets do the same min-max scaling for the wind data

    # In[18]:


    ### TODO calculate the minimum and maximum values of the 'Wind' column (2 lines of code)
    Wmin = weather['Wind'].min()
    Wmax = weather['Wind'].max()

    #print(Wmin, Wmax)

    ### TODO define a scale_wind function and apply it on the Wind column (~ 2-3 lines of code)
    def scale_wind(w):
        return (w - Wmin) / ( Wmax  - Wmin)

    ### TODO apply the scale_wind function on the 'Wind' column
    weather['Wind'] = weather['Wind'].applymap(scale_wind)


    # Let's inspect the table, the Temperature, Wind and Humidity columns should have values in the [0, 1] range.

    # In[19]:


    weather.describe()


    # ##### Dropping Columns
    #
    # The relative temperature column is correlated with the temperature, it will not give much extra information for the ML model. We want to remove this column from our `DataFrame`. We can do so using the `drop_column` method. Note that this method removes a column in-place - meaning that the `DataFrame` we act on will be modified.

    # In[20]:


    weather.drop_column('RTemp')


    # If we want to remove a column without modifying the original DataFrame, we can use the `drop` method. This method will return a new DataFrame without that column (or columns).

    # ##### Index
    #
    # Like `Series` objects, each `DataFrame` has an index attribute.

    # In[21]:




    # We can use the index values to subset the `DataFrame`. Lets use this to plot the first 48 values. Before plotting we have to transfer from the GPU memory to the system memory. We use the `to_array` method to return a copy of the data as a numpy array.

    # In[22]:


    selection = weather[weather.index<48]
    #plt.plot(selection['Hour'].to_array(), selection['Temperature'].to_array())
    #plt.xlabel('Hour')
    #plt.ylabel('Temperature [C]')


    # We can also change the index. Our dataset has one entry for each hour, so one could set the 'Hour' coulmn as index by calling
    # ```
    # weather = weather.set_index('Hour')
    # ```
    #
    # We do not perform this change now.

    # In[ ]:


    #weather = weather.set_index('Hour')


    # ### 2.2 Prepare bike sharing data
    # We start by downloading the data

    # In[24]:


    files = ['data/2011-capitalbikeshare-tripdata.csv', 'data/2012Q1-capitalbikeshare-tripdata.csv', 'data/2012Q2-capitalbikeshare-tripdata.csv', 'data/2012Q3-capitalbikeshare-tripdata.csv', 'data/2012Q4-capitalbikeshare-tripdata.csv']


    # Let's read the first file to have an idea of the dataset

    # In[25]:


    #cudf.read_csv(files[0])


    # We are only interested in the events when a bicicle was rented. Let us read the first column from all files, by specifying the `usecols` argument to [read_csv](https://docs.rapids.ai/api/cudf/nightly/api.html#cudf.io.csv.read_csv). We can use the `parse_dates` argument to parse the date string into a datetime variable, or the [to_datetime](https://docs.rapids.ai/api/cudf/nightly/api.html#cudf.to_datetime) function that we have used for the weather dataset. After all the tables are read we will concatenate them.
    #
    # Note: one has to specify a list of columns [ column1, column2 ] for the `usecol` argument.

    # In[26]:


    def read_bike_data(files):
        # Reads a list of files and concatenates them
        tables = []
        for filename in files:
            ### TODO read column 1 ('Start date') from the CSV file, and convert it to datetime format
            ### (1-2 lines of code)
            tmp_df = cudf.read_csv(filename, usecols=[1])

            ### END TODO
            tables.append(tmp_df)

        merged_df = cudf.concat(tables, ignore_index=True)

        # # Sanity checks
        # if merged_df.columns != ['Start date']:
        #     raise ValueError("Error incorrect set of columns read")
        # if merged_df['Start date'].dtype != 'datetime64[ns]':
        #     raise TypeError("Stard date should be converted to datetime type")

        return merged_df


    # In[27]:


    t_file_start = timer()
    bikes_raw = read_bike_data(files)
    t_file += timer()-t_file_start

    bikes_raw['Start date'] = cudf.to_datetime(bikes_raw['Start date'])


    # We want to count the number of rental events in every hour. We will define a new feature where we remove the minutes and seconds part of the time stamp. Since pandas has a convenient `floor` function defined to do it, we will convert the column to a pandas Series, transform it with the floor operation, and then put it back on the GPU.

    # In[28]:


    bikes_raw['Hour'] = bikes_raw['Start date'].to_pandas().dt.floor('h')


    # We will aggregate the number of bicicle rental events for each hour. We use the [groupby](https://docs.rapids.ai/api/cudf/nightly/api.html#groupby) function.

    # In[44]:


    bikes = bikes_raw.groupby('Hour').agg('count')
    bikes.columns = ['cnt']
    bikes.head(5)


    # In[32]:


   # bikes_raw_pd = bikes_raw.to_pandas()


    # In[45]:


    #bikes_pd = bikes_raw_pd.groupby('Hour').agg('count')
    #bikes_pd.columns = ['cnt']
    #bikes_pd.head(5)


    # Let's add a column to the new dataset: the date without the time of the day. We can derive that similarly to the 'Hour' feature above. After the groupby operation, the 'Hour' became the index of the dataset, we will apply the `floor` operation on the index.

    # In[64]:


    bikes['date'] = bikes.index.to_pandas().floor('D')


    # It will be usefull to define a set of additional features: hour of the day, day of month, month and year https://docs.rapids.ai/api/cudf/nightly/api.html#datetimeindex

    # In[53]:


    bikes['hr'] = bikes.index.hour

    ### TODO add year and month features (~ 2 lines of code)
    bikes['year'] = bikes.index.year
    bikes['month'] = bikes.index.month


    # #### Visualize data
    # It is a good practice to visulize the data. We will have to use the to_array() method to convert the cudF Series objects to numpy arrays that can be plotted.

    # In[56]:


    #plt.plot(bikes.index.to_array(), bikes['cnt'].to_array())


    # It is hard to see much apart from the global trend. Let's have a look how the 'cnt' variable looks like as a function the 'month' and 'hr' features. We will use [boxplot](https://seaborn.pydata.org/generated/seaborn.boxplot.html) from the Seaborn package.

    # In[57]:


    #fig, axes = plt.subplots(nrows=1,ncols=2)
    #fig.set_size_inches(12, 5)
    #sns.boxplot(data=bikes.to_pandas(), y="cnt",x="month",orient="v",ax=axes[0])
    #sns.boxplot(data=bikes.to_pandas(), y="cnt",x="hr",orient="v",ax=axes[1])
    #axes[0].set(xlabel='Months', ylabel='Count',title="Box Plot On Count Across months")
    #axes[1].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")
    #plt.show()


    # #### 3.2.1 Combine weather data with bike rental data

    # In[65]:


    gdf_bw = bikes.merge(weather, left_index=True, right_on='Hour', how='inner')

    # inspect the merged table
    #gdf_bw


    # We can see that the data is not sorted after the merge use the [sort_values](https://docs.rapids.ai/api/cudf/nightly/api.html#cudf.core.dataframe.DataFrame.sort_values) method to

    # In[66]:


    ### TODO sort the table according to the index (1 line of code)
    gdf_bw = gdf_bw.sort_values(by='Hour')

    # Inspect the sorted table
    #gdf_bw


    # ### 3.3 Add working day feature
    #
    # Apart from the weather, in important factor that influences people's daily activities is whether it is a working day or not. In this section we will create a working day feature. First we add the weekday as a new feature column.
    # We first
    # We can use the [weekday](https://docs.rapids.ai/api/cudf/nightly/api.html#cudf.core.series.DatetimeProperties.weekday) attribute of the [datetime](https://docs.rapids.ai/api/cudf/nightly/api.html#datetimeindex)

    # In[67]:


    gdf_bw['Weekday'] = gdf_bw['date'].dt.weekday


    # Next create a table with all the holidays in Washington DC in 2011-2011

    # In[68]:


    holidays = cudf.DataFrame({'date': ['2011-01-17', '2011-02-21', '2011-04-15', '2011-05-30', '2011-07-04', '2011-09-05', '2011-11-11', '2011-11-24', '2011-12-26', '2012-01-02', '2012-01-16', '2012-02-20', '2012-04-16', '2012-05-28', '2012-07-04', '2012-09-03', '2012-11-12', '2012-11-22', '2012-12-25'],
    'Description': ["Martin Luther King Jr. Day", "Washington's Birthday", "Emancipation Day", "Memorial Day", "Independence Day", "Labor Day", "Veterans Day", "Thanksgiving", "Christmas Day",
    "New Year's Day", "Martin Luther King Jr. Day", "Washington's Birthday", "Emancipation Day", "Memorial Day", "Independence Day", "Labor Day", "Veterans Day", "Thanksgiving", "Christmas Day"]})

    # Print the dataframe
    #holidays


    # We convert the date from string to datetime type, and drop the description column. Additionally we add a new column marked 'Holiday'. This will be useful to mark the holidays after we merge the tables.

    # In[69]:


    holidays['date'] = cudf.to_datetime(holidays['date'])
    holidays.drop_column('Description')
    holidays['Holiday'] = 1
    #holidays


    # Now we are ready to merge the tables.

    # In[76]:


    ### TODO merge tables and on the column 'date', use a left merge
    gdf = gdf_bw.merge(holidays, on='date', how='left')

    # inspect the result
    #gdf


    # We reset the index to 'Hour' and sort the table accordingly. Notice that most of the rows in the 'Holiday' column are filled with `<NA>`, only the dates that appeared in the holiday table are filled with 1. We shall fill the empty fields with zero.

    # In[77]:


    gdf = gdf.set_index('Hour')
    gdf = gdf.sort_index()

    ### TODO fill empty holiday values with zero
    gdf['Holiday'] = gdf['Holiday'].fillna(0)
    #gdf


    # Next, we create a workingday feature. One could do that simply with the following operation.
    # ```
    # gdf['Workingday'] = (gdf['Weekday'] < 5) & (gdf['Holiday']!=1)
    # ```
    # But we could do it with user defined functions too. Previously we have only used UDF to process elements of a series. Now we will process rows of a dataframe and
    # combine the 'Weekday' and 'Holiday' columns to calculate the now feature 'Workingday'

    # In[78]:


    def workday_kernel(Weekday, Holiday, Workingday):
        for i, (w, h) in enumerate(zip(Weekday, Holiday)):
            Workingday[i] = w < 5 and h != 1


    # In[79]:


    gdf = gdf.apply_rows(workday_kernel, incols=['Weekday', 'Holiday'], outcols=dict(Workingday=np.float64), kwargs=dict())


    # More on user defined functions in our [blog](https://medium.com/rapids-ai/user-defined-functions-in-rapids-cudf-2d7c3fc2728d) and in the [documentation](https://docs.rapids.ai/api/cudf/nightly/guide-to-udfs.html).
    #
    # https://numba.pydata.org/numba-doc/dev/reference/pysupported.html

    # After this step we will not need the 'Holiday' and 'date' columns, we can drop them

    # In[80]:


    gdf = gdf.drop(['Holiday', 'date'])


    # ### 2.4 One-hot encoding
    #
    # We have all now the data in a single table, but we still want to change their encoding. We're going to create one-hot encoded variables, also known as dummy variables, for each of the time variables as well as the weather situation.
    #
    #
    # A summary from https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/:
    #
    # "The integer values have a natural ordered relationship between each other and machine learning algorithms may be able to understand and harness this relationship.
    # For categorical variables where no such ordinal relationship exists, the integer encoding is not enough.
    #
    # In fact, using this encoding and allowing the model to assume a natural ordering between categories may result in poor performance or unexpected results (predictions halfway between categories).
    #
    # In this case, a one-hot encoding can be applied to the integer representation. This is where the integer encoded variable is removed and a new binary variable is added for each unique integer value.
    # "
    #
    # We start by one-hot encoding the 'Weather' column using the [one_hot_encoding](https://docs.rapids.ai/api/cudf/nightly/api.html#cudf.core.dataframe.DataFrame.one_hot_encoding) method from cuDF DataFrame. This is very the [get_dummies](https://docs.rapids.ai/api/cudf/nightly/api.html#cudf.core.reshape.get_dummies) function (which might be more familiar for Pandas users), but one_hot_encoding works on a single input column and performs the operation in place.

    # In[81]:


    codes = gdf['Weather'].unique()
    gdf = gdf.one_hot_encoding('Weather', 'Weather_dummy', codes)
    # Inspect the results
    #gdf.head(3)


    # We're going to drop the original variable as well as one of the new dummy variables so we don't create colinearity (more about this problem [here](https://towardsdatascience.com/one-hot-encoding-multicollinearity-and-the-dummy-variable-trap-b5840be3c41a)).

    # In[82]:


    gdf = gdf.drop(['Weather', 'Weather_dummy_1'])


    # We create a copy of the dataset. It will make it easier to start over in case something would go wrong during the next excercise.

    # In[83]:


    gdf_backup = gdf.copy()


    # In[85]:


    dummies_list = ['month', 'hr', 'Weekday']

    gdf = gdf_backup.copy()

    for item in dummies_list:
        ### Todo implement one-hot encoding for item
        codes = gdf[item].unique()
        gdf = gdf.one_hot_encoding(item, item + '_dummy', codes)
        gdf = gdf.drop('{}_dummy_1'.format(item))
        gdf = gdf.drop(item) # drop the original item


    gdf['year'] = gdf['year'] - 2011
    # gdf.drop_column('year') # will need leater for train-test spliw


    # ### 2.5 Save the prepared dataset

    # In[91]:


    t_file_start = timer()
    gdf.to_csv('data/bike_sharing.csv')
    t_file += timer() - t_file_start

    # ## 3. Predict bike rentals with cuML
    #
    # cuML is a GPU accelerated machine learning library. cuML's Python API mirrors the [Scikit-Learn](https://scikit-learn.org/stable/) API.
    #
    # cuML currently requires all data be of the same type, so this loop converts all values into floats

    # In[92]:

    t_cudf_stop = timer()




    # In[93]:

    t_cuml_start = t_cudf_stop

    for col in gdf.columns:
        gdf[col] = gdf[col].astype('float64')


    # ### 3.1 Prepare training and test data
    # It is customary to denote the input feature matrix with X, and the target that we want to predict with y. We separete the target column 'cnt' from the rest of the table.

    # In[94]:


    y = gdf['cnt']
    X = gdf.drop('cnt')


    # Let's split the data randomly into a train and a test set

    # In[95]:


    X_train, X_test, y_train, y_test = cuml.preprocessing.model_selection.train_test_split(X, y)

    #test = gdf.query('yr == 1') #.drop(dummies_list)
    #train = gdf.query('yr == 0') #.drop(dummies_list)


    # ### 3.2 Linear regression

    # In[111]:


    reg = cuml.LinearRegression()
    reg.fit(X_train, y_train)


    # In[129]:


    #X_train_np = X_train.as_matrix() #to_pandas().to_numpy()
    #y_train_np = y_train.to_array()
    #X_test_np = X_test.as_matrix() #to_pandas().to_numpy()
    #y_test_np = y_test.to_array()


    # In[100]:


    #import sklearn


    # In[109]:


    #reg_skl = sklearn.linear_model.LinearRegression()\nreg_skl.fit(X_train_np, y_train_np)


    # We can make prediction with the trained data

    # In[ ]:


    y_hat = reg.predict(X_test)


    # We can visualize the how well the model works. Let's plot data for may 2012:

    # In[112]:


    # In[114]:


    train_score = reg.score(X_train, y_train)
    ### TODO calculate test score (the score on X_test, ~ 1 line of code)
    test_score = reg.score(X_test, y_test)

    #print('train score', train_score)
    #print('test score', test_score)


    # ### 3.3  Save and load the trained model
    # We can pickle any cuML model

    # In[115]:


    t_file_start = timer()
    pickle_file = 'my_model.pickle'

    with open(pickle_file, 'wb') as pf:
        pickle.dump(reg, pf)


    # Load the saved model

    # In[116]:


    with open(pickle_file, 'rb') as pf:
            loaded_model = pickle.load(pf)

    t_file_cuml = timer() - t_file_start

    #print('Loaded model   score', loaded_model.score(X_test, y_test))
    #print('Original model score', reg.score(X_test, y_test))


    # ### 3.4 Ridge regression with hyperparameter tuning
    # We're going to do a small hyperparameter search for alpha, checking 100 different values. This is fast to do with RAPIDS. Also notice that we are appending the results of each Ridge model onto the dictionary containing our earlier results, so we can more easily see which model is the best at the end.

    # In[117]:


    output = {'score_OLS': test_score}

    for alpha in np.arange(0.01, 1, 0.01): #alpha value has to be positive
        ridge = cuml.Ridge(alpha=alpha, fit_intercept=True)
        ### TODO fit the model and calculate the test score (2 lines of code)
        ridge.fit(X_train, y_train)
        score = ridge.score(X_test, y_test)
        ### END EXCERCISE ###
        output['score_RIDGE_{}'.format(alpha)] = score


    # Here we see that our regulaized model does better than the rest, include OLS with all the variables.

    # In[118]:


    #print('Max score: {}'.format(max(output, key=output.get)))


    # ### 3.5 Additional cuML models (Optional)
    # #### 3.5.1 Support vector regression

    # In[127]:


    # reg = cuml.svm.SVR(kernel='rbf', gamma=0.1, C=100, epsilon=0.1)
    # reg.fit(X_train, y_train)
    # reg.score(X_train, y_train)
    # reg.score(X_test, y_test)


    # In[130]:


    #reg = sklearn.svm.SVR(kernel='rbf', gamma=0.1, C=100, epsilon=0.1)
    #reg.fit(X_train_np, y_train_np)
    #reg.score(X_train_np, y_train_np)
    #reg.score(X_test_np, y_test_np)


    # #### 3.5.2 KNN Regression

    # In[134]:


    knn = cuml.neighbors.KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train, convert_dtype=True)
    pred = knn.predict(X_test)
    knn.score(X_test, y_test)

    #knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
    #knn.fit(X_train_np, y_train_np,)
    #pred = knn.predict(X_test_np)
    #knn.score(X_test_np, y_test_np)




    t_stop = timer()
    t_cudf = t_cudf_stop - t_start
    t_cuml = t_stop - t_cuml_start #- t_file_cuml
    return t_cudf, t_cuml, t_file, t_file_cuml

# warmup
run_everything()

# benchmark
n_loop = 5
t_cudf = np.zeros(n_loop)
t_cuml = np.zeros(n_loop)
t_file = np.zeros(n_loop)
t_file_cuml = np.zeros(n_loop)

for i in range(5):
    t_cudf[i], t_cuml[i], t_file[i], t_file_cuml[i] = run_everything()
    print('t_file', t_file[i], 't_cudf', t_cudf[i], "t_cuml:", t_cuml[i], "t_cuml_file", t_file_cuml[i])

print("Summary (mean and std):")
print("t_file", t_file.mean(), t_file.std())
print("t_cudf", t_cudf.mean(), t_cudf.std())
print("t_cuml", t_cuml.mean(), t_cuml.std())
