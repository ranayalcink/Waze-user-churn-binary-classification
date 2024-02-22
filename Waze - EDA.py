#!/usr/bin/env python
# coding: utf-8

# # **Waze Project - Exploratory Data Analysis**
# 

# The Waze data analytics team is still in the early stages of their user churn project. To get clear insights, the user data must be inspected and prepared for the upcoming process of exploratory data analysis (EDA).

# ### Inspect and analyze data
# 
# **The purpose** of this project is to investigate and understand the data provided.
#  
# **The goal** is to use a dataframe contructed within Python, perform a cursory inspection of the provided dataset, and inform team members of your findings.

# In[1]:


# Import packages for data manipulation
### YOUR CODE HERE ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load dataset into dataframe
df = pd.read_csv('waze_dataset.csv')


# In[3]:


df.to_csv('waze.csv')


# In[4]:


### YOUR CODE HERE ###
df.head(10)


# In[5]:


### YOUR CODE HERE ###
df.info()


# - There is no missing data in the first 10 rows.
# - According to df.info() output, there are different types of data in our dataset such as integer, float, and object. The dataset has 13 columns and 14999 rows. 
# - Only column that has missing values is label. there are 700 missing values in label column.

# In[6]:


# Isolate rows with null values
### YOUR CODE HERE ###
null_values = df[df['label'].isnull()]
# Display summary stats of rows with null values
### YOUR CODE HERE ###
null_values.describe()


# In[7]:


# Isolate rows without null values
### YOUR CODE HERE ###
no_null_values = df[~df['label'].isnull()]
# Display summary stats of rows without null values
### YOUR CODE HERE ###
no_null_values.describe()


# - There are no significant difference between the datasets other than count. Mean and std seems pretty consistent. 

# In[8]:


# Get count of null values by device
### YOUR CODE HERE ###

null_values['device'].value_counts()


# - While 447 Iphone users had null values, the number of Android users who had null values is 253. 

# In[9]:


# Calculate % of iPhone nulls and Android nulls
### YOUR CODE HERE ###
null_values['device'].value_counts(normalize=True)


# In[10]:


# Calculate % of iPhone users and Android users in full dataset
### YOUR CODE HERE ###
df['device'].value_counts(normalize=True)


# The percentage of missing values by each device is consistent with their representation in the data overall.
# 
# There is nothing to suggest a non-random cause of the missing data.

# In[11]:


# Calculate counts of churned vs. retained
### YOUR CODE HERE ###
print(df['label'].value_counts())
df['label'].value_counts(normalize=True)


# This dataset contains 82% retained users and 18% churned users.

# In[12]:


# Calculate median values of all columns for churned and retained users
### YOUR CODE HERE ###
df.groupby('label').median()


# This offers an interesting snapshot of the two groups, churned vs. retained:
# 
# Users who churned averaged ~3 more drives in the last month than retained users, but retained users used the app on over twice as many days as churned users in the same time period.
# 
# The median churned user drove ~200 more kilometers and 2.5 more hours during the last month than the median retained user.
# 
# It seems that churned users had more drives in fewer days, and their trips were farther and longer in duration. 

# In[13]:


# Group data by `label` and calculate the medians
### YOUR CODE HERE ###
medians_by_label = df.groupby('label').median(numeric_only=True)
# Divide the median distance by median number of drives
### YOUR CODE HERE ###
medians_by_label['driven_km_drives']/medians_by_label['drives']


# The median user from both groups drove ~73 km/drive.

# In[14]:


# Divide the median distance by median number of driving days
### YOUR CODE HERE ###
medians_by_label['driven_km_drives']/medians_by_label['driving_days']


# In[15]:


# Divide the median number of drives by median number of driving days
### YOUR CODE HERE ###
medians_by_label['drives']/medians_by_label['driving_days']


# The median user who churned drove 608 kilometers each day they drove last month, which is almost 250% the per-drive-day distance of retained users. The median churned user had a similarly disproporionate number of drives per drive day compared to retained users.

# In[16]:


# For each label, calculate the number of Android users and iPhone users
### YOUR CODE HERE ###
df.groupby(['label','device']).size()


# In[17]:


# For each label, calculate the percentage of Android users and iPhone users
### YOUR CODE HERE ###
df.groupby('label')['device'].value_counts(normalize=True)


# The ratio of iPhone users and Android users is consistent between the churned group and the retained group, and those ratios are both consistent with the ratio found in the overall dataset.

# 1. The data contains missing values only in 'label' column. There are 700 values missing in that column. There was no pattern to the missing data.
# 
# 2. The mean value gets affected by the outliers in the data. Median doesn't get affected by outliers therefore it can be a more accurate measure to estimate data.
# 
# 3. In the analysis, it is observed that churned users has driven 608 km per day. It is a significantly higher value compared to retained users who has driven 245 km per day. Waze team should explore the reasons behind why churned users use the app or their specific needs and problems. 
# 
# 4. The iPhone users is represented by % 65 of the data while Android users had % 35 approximately.
# 
# 5. Churned users generally drove longer distances in fewer days.

# 1. All the columns except the 'ID' column are applicable to the problem since they represent user behavior.
# 2. We can eliminate the 'ID' column because we are not interested in a particular user.
# 3. We can use info() method to calculate the count of non-null values.
#    - Request from data owners to fill the missing values
#    - If the count of missing values is low and the missing values has low influence on the data, delete the rows with missing values
#    - Create a NaN category
#    - Fill missing values with median or average
# 4. Any data point that falls beyond 1.5 times the interquartile range is considered an outlier. We can use boxplot to visualize the outliers. To handle outliers:
#    - Delete them
#    - Reassign them
#    - Leave them

# In[18]:


# sessions box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['sessions'])
plt.title('sessions boxplot')
plt.show()


# In[19]:


# Histogram
sns.histplot(x=df['sessions'])
plt.axvline(df['sessions'].median(), color='red', linestyle='--')
plt.text(75,1200,'median=56', color='red', backgroundcolor='bisque')
plt.title('sessions histogram')
plt.show()


# The `sessions` variable is a right-skewed distribution with half of the observations having 56 or fewer sessions. However, as indicated by the boxplot, some users have more than 700.

# In[20]:


# drives box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['drives'])
plt.title('drives boxplot')
plt.show()


# In[21]:


# drives histogram
sns.histplot(x=df['drives'])
plt.axvline(df['drives'].median(), color='red', linestyle='--')
plt.text(75,1000,'median=48',color='red', backgroundcolor='bisque')
plt.title('drives histogram')
plt.show()


# The `drives` information follows a distribution similar to the `sessions` variable. It is right-skewed, approximately log-normal, with a median of 48. However, some drivers had over 400 drives in the last month.

# In[22]:


# total_sessions Box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['total_sessions'])
plt.title('total_sessions boxplot')
plt.show()


# In[23]:


# total_sessions Histogram
sns.histplot(x=df['total_sessions'])
plt.axvline(df['total_sessions'].median(), color='red', linestyle='--')
plt.text(230,700,'median=159.6',color='red', backgroundcolor='bisque')
plt.title('total_sessions histogram')
plt.show()


# The `total_sessions` is a right-skewed distribution. The median total number of sessions is 159.6. This is interesting information because, if the median number of sessions in the last month was 48 and the median total sessions was ~160, then it seems that a large proportion of a user's total drives might have taken place in the last month. This is something you can examine more closely later.

# In[24]:


# n_days_after_onboarding Box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['n_days_after_onboarding'])
plt.title('n_days_after_onboarding boxplot')
plt.show()


# In[25]:


# n_days_after_onboarding Histogram
sns.histplot(x=df['n_days_after_onboarding'])
plt.title('n_days_after_onboarding histogram')
plt.show()


# The total user tenure (i.e., number of days since onboarding) is a uniform distribution with values ranging from near-zero to \~3,500 (\~9.5 years).

# In[26]:


# driven_km_drives Box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['driven_km_drives'])
plt.title('driven_km_drives')
plt.show()


# In[27]:


# driven_km_drives Histogram
sns.histplot(x=df['driven_km_drives'])
plt.axvline(df['driven_km_drives'].median(), color='red', linestyle='--')
plt.text(4000,700,'median=3495',color='red',backgroundcolor='bisque')
plt.title('driven_km_drives histogram')
plt.show()


# The number of drives driven in the last month per user is a right-skewed distribution with half the users driving under 3,495 kilometers. As you discovered in the analysis from the previous course, the users in this dataset drive _a lot_. The longest distance driven in the month was over half the circumferene of the earth.

# In[28]:


# duration_minutes_drives Box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['duration_minutes_drives'])
plt.title('duration_minutes_drives')
plt.show()


# In[29]:


# duration_minutes_drives Histogram
sns.histplot(x=df['duration_minutes_drives'])
plt.axvline(df['duration_minutes_drives'].median(), color='red', linestyle='--')
plt.text(2000,700,'median=1478',color='red',backgroundcolor='bisque')
plt.title('duration_minutes_drives histogram')
plt.show()


# The `duration_minutes_drives` variable has a heavily skewed right tail. Half of the users drove less than \~1,478 minutes (\~25 hours), but some users clocked over 250 hours over the month.

# In[30]:


# activity_days Box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['activity_days'])
plt.title('activity_days')
plt.show()


# In[31]:


# activity_days Histogram
sns.histplot(x=df['activity_days'])
plt.title('activity_days histogram')
plt.show()


# Within the last month, users opened the app a median of 16 times. The box plot reveals a centered distribution. The histogram shows a nearly uniform distribution of ~500 people opening the app on each count of days. However, there are ~250 people who didn't open the app at all and ~250 people who opened the app every day of the month.
# 
# This distribution is noteworthy because it does not mirror the `sessions` distribution, which you might think would be closely correlated with `activity_days`.

# In[32]:


# driving_days Box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['driving_days'])
plt.title('driving_days')
plt.show()


# In[33]:


# driving_days Histogram
sns.histplot(x=df['driving_days'])
plt.title('driving_days histogram')
plt.show()


# The number of days users drove each month is almost uniform, and it largely correlates with the number of days they opened the app that month, except the `driving_days` distribution tails off on the right.
# 
# However, there were almost twice as many users (\~1,000 vs. \~550) who did not drive at all during the month. This might seem counterintuitive when considered together with the information from `activity_days`. That variable had \~500 users opening the app on each of most of the day counts, but there were only \~250 users who did not open the app at all during the month and ~250 users who opened the app every day. Flag this for further investigation later.

# In[34]:


# device Pie chart
device_counts = df['device'].value_counts()
plt.pie(device_counts, labels=device_counts.index, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.
plt.show()


# There are nearly twice as many iPhone users as Android users represented in this data.

# In[35]:


# label Pie chart
data = df['label'].value_counts()
plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90)
plt.show()


# Less than 18% of the users churned.

# In[36]:


# Histogram
plt.figure(figsize=(10,3))
label=['driving_days', 'activity_days']
plt.hist([df['driving_days'], df['activity_days']],
         bins=range(0,33),
         label=label)
plt.xlabel('days')
plt.ylabel('count')
plt.legend()
plt.title('driving_days vs. activity_days')
plt.show()


# In[37]:


# Scatter plot
plt.scatter(df['driving_days'],df['activity_days'], s=11, alpha=0.2)
plt.plot([0,31],[0,31],linestyle='--',color='red')
plt.xlabel('driving_days')
plt.ylabel('activity_days') 
plt.title('driving_days vs. activity_days')
plt.show()


# In[38]:


# Histogram
plt.figure(figsize=(5,3))
sns.histplot(data=df,
             x=df.device,
             hue=df.label,
             multiple='dodge',
             shrink=0.9)
plt.title('Retention by device')
plt.xlabel('')
plt.show()


# The proportion of churned users to retained users is consistent between device types.

# In[39]:


df['km_per_driving_day'] = df['driven_km_drives']/df['driving_days']

df['km_per_driving_day'].describe()


# In[40]:


df.loc[df['km_per_driving_day']==np.inf, 'km_per_driving_day'] = 0
df['km_per_driving_day'].describe()


# The maximum value is 15,420 kilometers _per drive day_. This is physically impossible. Driving 100 km/hour for 12 hours is 1,200 km. It's unlikely many people averaged more than this each day they drove, so, for now, disregard rows where the distance in this column is greater than 1,200 km.

# In[41]:


# Histogram
plt.figure(figsize=(10,3))
mask = df['km_per_driving_day']<1200
sns.histplot(x=df['km_per_driving_day'][mask],
             hue=df['label'],
             multiple='fill')
plt.ylabel('%',rotation=0)
plt.title('Churn rate by mean km per driving day')
plt.show()


# The churn rate tends to increase as the mean daily distance driven increases, confirming what was found in the previous course. It would be worth investigating further the reasons for long-distance users to discontinue using the app.

# In[42]:


# Histogram
sns.histplot(x=df['driving_days'],
             bins=range(0,32),
             hue=df['label'],
             multiple='fill')
plt.title('Churn rate by number of driving days')
plt.ylabel('%',rotation=0)
plt.show()


# The churn rate is highest for people who didn't use Waze much during the last month. The more times they used the app, the less likely they were to churn. While 40% of the users who didn't use the app at all last month churned, nobody who used the app 30 days churned.

# In[43]:


df['percent_sessions_in_last_month'] = df['sessions']/df['total_sessions']


# In[44]:


# Histogram
sns.histplot(x=df['percent_sessions_in_last_month'],
             hue=df['label'])
plt.axvline(df['percent_sessions_in_last_month'].median(),linestyle='--',color='red')
plt.title('percent_sessions_in_last_month histogram')
plt.show()


# Half of the people in the dataset had 40% or more of their sessions in just the last month, yet the overall median time since onboarding is almost five years.

# In[45]:


# Histogram
sns.histplot(x=df['n_days_after_onboarding'][df['percent_sessions_in_last_month']>=0.4])
plt.title('n_days_after_onboarding for users with >=40% sessions in last month histogram')
plt.show()


# The number of days since onboarding for users with 40% or more of their total sessions occurring in just the last month is a uniform distribution. This is very strange. It's worth asking Waze why so many long-time users suddenly used the app so much in the last month.

# In[46]:


def outlier_imputer(column,percentile):
    threshold = df[column].quantile(percentile)
    df.loc[df[column] > threshold, column] = threshold
    return print(f'{column} threshold: {threshold:.2f}')


# In[47]:


columns = ['sessions','drives','total_sessions','driven_km_drives','duration_minutes_drives']
for i in columns:
    outlier_imputer(i,0.95)


# In[48]:


df.to_csv('/Users/eniseranabeklen/Desktop/portfolio/waze_cleaned')


# ## Conclusion
# 
# 1. The data contains missing values only in 'label' column. There are 700 values missing in that column. There was no pattern to the missing data.
# 
# 2. The mean value gets affected by the outliers in the data. Median doesn't get affected by outliers therefore it can be a more accurate measure to estimate data.
# 
# 3. In the analysis, it is observed that churned users has driven 608 km per day. It is a significantly higher value compared to retained users who has driven 245 km per day. Waze team should explore the reasons behind why churned users use the app or their specific needs and problems. 
# 
# 4. The iPhone users is represented by % 65 of the data while Android users had % 35 approximately.
# 
# 5. Churned users generally drove longer distances in fewer days.
# 
# 6. Most of the columns have uniform or right-skewed distributions. Right-skewed distribution means that most users had values in the left side of the range. Uniform distribution means the users have equally distributed within the range. 
# 
# 7. The data was not problematic generally. However 'driven_km_drives' column has some unrealistic outlier values. Additionaly some data points showed that driving_days has higher value than activity days which is impossible.
# 
# 8. Waze team should confirm the maximum number of driving days and activity days which should be the same. Also I want to ask that why so many users started the use the app in the last month even though they signed up way earlier.
# 
# 9. %17.7 of users are churned.
# 
# 10. Distance driven per driving day had a positive correlation with user churn. The farther a user drove on each driving day, the more likely they were to churn. On the other hand, number of driving days had a negative correlation with churn. Users who drove more days of the last month were less likely to churn.
# 
# 11. Users of all tenures from brand new to ~10 years were relatively evenly represented in the data. This is borne out by the histogram for n_days_after_onboarding, which reveals a uniform distribution for this variable.
# 
