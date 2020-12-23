#!/usr/bin/env python
# coding: utf-8

# # Crime Data from St. Louis Missouri (2008-2015)
# ## Loading and Cleaning the Data

# In[1]:


import gmaps
import numpy as np
import pandas as pd
import requests
import time
import matplotlib.pyplot as plt
from scipy.stats import linregress


# In[2]:


# Prepare list of files to load.
load_urls = ['2008_data.csv', '2009.csv', '2010.csv', '2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv']

# Create list of columns we want to use in the final DF.
column_names = ['CADAddress', 'CADStreet', 'CodedMonth', 'Count', 'Crime', 'ShortCrimeCode', 'UCRType',                'UCRCrime', 'DateOccured', 'Description', 'District', 'FlagCrime', 'FlagUnfounded',                'ILEADSAddress','ILEADSStreet', 'LocationComment', 'LocationName', 'Neighborhood',                'NeighborhoodName', 'NeighborhoodPrimaryDistrict', 'NeighborhoodAddlDistrict', 'Latitude',                'Longitude', 'Year']


# In[3]:


# Create empty DF for crime data
crime_df = pd.DataFrame(columns = column_names)

# Read single year crime files and store into final data frame
for file in load_urls:
    load_df = pd.read_csv(f'Resources/{file}')
    
    # Select only the columns we want from the single year csv.
    load_df = load_df[column_names]
    
    # Append data from the single year csv into the final DF
    crime_df = crime_df.append(load_df,ignore_index=True)


# In[4]:


# Preview data
crime_df.head()


# In[5]:


# How many records do we have?
crime_df['CADAddress'].count()


# In[6]:


# Remove rows that do not have location information complete.
crime_df = crime_df.loc[(pd.isna(crime_df['Latitude'])==False) & (pd.isna(crime_df['Longitude'])==False) & (pd.isna(crime_df['NeighborhoodName'])==False)]


# In[7]:


# How many records do we have after dropping nulls?
crime_df['CADAddress'].count()


# In[8]:


# I had issues with indexing into the list created by the split string funtion for reach row so I created a new column for the
## split string then did a for loop to assign the year from that column to the year column if the year column is null.

crime_df['Year_calc'] = crime_df['CodedMonth'].str.split('-')

for index in crime_df.index:
    if pd.isna(crime_df['Year'][index]) == True:
        crime_df['Year'][index] =  crime_df['Year_calc'][index][0]
crime_df.head()


# In[9]:


# Remove the no longer needed Year_calc column
crime_df.drop('Year_calc', inplace=True, axis=1)


# In[10]:


# Change the data type of the year and count column to be an int to remove the decimal
crime_df['Year'] = crime_df['Year'].astype('int')
crime_df['Count'] = crime_df['Count'].astype('int')

crime_df.head()


# In[11]:


# Remove row where CADStreet is unknown
crime_df = crime_df.loc[crime_df['CADStreet'] != 'unknown 0000']


# In[12]:


crime_df


# ## Trends Per Year

# In[ ]:





# In[ ]:





# In[ ]:





# ## Offenses Per Neighborhood

# In[13]:


# Remove where the Neighborhood is unknown.
nh_crime = crime_df.loc[crime_df['NeighborhoodName']!= 'Unknown']

# Create a DF that is a simple groupby NeighborhoodName to reference in the overall analysis.
crime_nh = nh_crime.groupby(['Neighborhood', 'NeighborhoodName'])

# Create DF that holds the total count of crimes per neighborhood.
nh_tot = crime_nh['Count'].sum().reset_index()

# Take only the largest 20 crime counts (89 neighborhoods makes the graph impossible to read.)
nh_tot_top = nh_tot.nlargest(20,'Count')

# Plot as a bar chart.
plt.bar(nh_tot_top['NeighborhoodName'], nh_tot_top['Count'], color="b", align="center")
plt.xticks(nh_tot_top['NeighborhoodName'], nh_tot_top['NeighborhoodName'], rotation='vertical')

plt.xlabel('Neighborhood')
plt.ylabel('Total Crime Counts')

plt.title('Top 20 Neighborhoods with Highest Total Recorded Crime')

plt.savefig('Output/WorstNeighborhoods.png')

plt.show()


# In[14]:


# For the top 20 most "dangerous" Neighborhoods, rank by percent of violent crimes.
top_twenty = pd.merge(crime_df,nh_tot_top, on=['NeighborhoodName', 'Neighborhood'], how='right',suffixes=('', '_Total') )
tp_twenty = top_twenty.groupby(['NeighborhoodName', 'UCRType']).agg({'Count':'sum', 'Count_Total':'mean'}).reset_index()

tp_twenty['Pct'] = tp_twenty['Count']/top_twenty['Count_Total']
tp_twenty
# colors = {1:'red', 2:'blue'}
# c = tp_twenty['UCRType'].apply(lambda x: colors[x])

# # Plot as a bar chart.
# plt.bar(tp_twenty['NeighborhoodName'], tp_twenty['Count'], color=c, align="center")
# plt.xticks(tp_twenty['NeighborhoodName'], tp_twenty['NeighborhoodName'], rotation='vertical')

# plt.xlabel('Neighborhood')
# plt.ylabel('Total Crime Counts')

# plt.title('Top 10 Neighborhoods with Highest Total Recorded Crime')

# plt.savefig('Output/WorstNeighborhoods_UCRPct.png')

# plt.show()


# In[15]:


tp_twenty_pct = top_twenty.groupby(['UCRType']).agg({'Count':'sum'}).reset_index()
# top_twenty
explode = (0.1,0)

labels={1:'Severe', 2:'Non-Severe'}
l = tp_twenty_pct['UCRType'].apply(lambda x: labels[x])

plt.title('Top 20 Neighborhoods with Highest Total Recorded Crime')
plt.pie(tp_twenty_pct['Count'],explode=explode, labels=l,
        autopct="%1.1f%%", shadow=False, startangle=140)


# In[16]:


top_twenty


# In[17]:


tp_year = top_twenty.groupby(['NeighborhoodName', 'Year'])['Count'].sum().reset_index()

plt.plot(tp_year['Year'], tp_year['Count'], label="Crime Tends across Neighborhoods")


# In[18]:


tp_year_tot = top_twenty.groupby(['Year'])['Count'].sum().reset_index()

plt.title('Total Crime Reports per Year (Upper 20 Neighborhoods)')
plt.xlabel('Year')
plt.ylabel('Total Crime Counts')

plt.plot(tp_year_tot['Year'], tp_year_tot['Count'], label="Crime Tends across Neighborhoods")

plt.savefig('Output/WorstNeighborhoods_YrChg.png')

plt.show()


# In[19]:


# Take only the largest 20 crime counts (89 neighborhoods makes the graph impossible to read.)
nh_tot_bottom = nh_tot.nsmallest(20,'Count')

# Plot as a bar chart.
plt.bar(nh_tot_bottom['NeighborhoodName'], nh_tot_bottom['Count'], color="b", align="center")
plt.xticks(nh_tot_bottom['NeighborhoodName'], nh_tot_bottom['NeighborhoodName'], rotation='vertical')

plt.xlabel('Neighborhood')
plt.ylabel('Total Crime Counts')
plt.title('Top 20 Neighborhoods with Lowest Total Recorded Crime')

plt.savefig('Output/BestNeighborhoods.png')

plt.show()


# In[20]:


# For the top 20 least "dangerous" Neighborhoods, rank by percent of violent crimes.
bottom_twenty = pd.merge(crime_df,nh_tot_bottom, on=['NeighborhoodName', 'Neighborhood'], how='right',suffixes=('', '_Total') )
btm_twenty = bottom_twenty.groupby(['NeighborhoodName', 'UCRType']).agg({'Count':'sum', 'Count_Total':'mean'}).reset_index()

colors = {1:'red', 2:'blue'}
c = btm_twenty['UCRType'].apply(lambda x: colors[x])

# Plot as a bar chart.
plt.bar(btm_twenty['NeighborhoodName'], btm_twenty['Count'], color= c, align="center") #bottom_twenty['UCRType'].map(colors), align="center")
plt.xticks(btm_twenty['NeighborhoodName'], btm_twenty['NeighborhoodName'], rotation='vertical')

plt.xlabel('Neighborhood')
plt.ylabel('Total Crime Counts')

plt.title('Top 10 Neighborhoods with Lowest Total Recorded Crime')

plt.savefig('Output/BestNeighborhoods_UCRPct.png')

plt.show()


# In[21]:


bottom_twenty_pct = btm_twenty.groupby(['UCRType']).agg({'Count':'sum'}).reset_index()

explode = (0.1,0)

labels={1:'Severe', 2:'Non-Severe'}
l = bottom_twenty_pct['UCRType'].apply(lambda x: labels[x])

plt.pie(bottom_twenty_pct['Count'],explode=explode, labels=l,
        autopct="%1.1f%%", shadow=False, startangle=140)
plt.title('Top 20 Neighborhoods with Lowest Total Recorded Crime')

plt.savefig('Output/WorstNeighborhoods_UCRPie.png')


# In[22]:


bottom_twenty_pct


# In[23]:


btm_year = bottom_twenty.groupby(['NeighborhoodName', 'Year'])['Count'].sum().reset_index()

plt.plot(btm_year['Year'], btm_year['Count'], label="Crime Tends across Neighborhoods")


# In[24]:


btm_year_tot = bottom_twenty.groupby(['Year'])['Count'].sum().reset_index()

plt.title('Total Crime Reports per Year (Lower 20 Neighborhoods)')
plt.xlabel('Year')
plt.ylabel('Total Crime Counts')

plt.plot(btm_year_tot['Year'], btm_year_tot['Count'], label="Crime Tends across Neighborhoods")

plt.savefig('Output/BestNeighborhoods_YrChg.png')

plt.show()


# In[25]:


crime_nh = nh_crime.groupby(['NeighborhoodName','UCRCrime'])
nh_offense = crime_nh['Count'].sum().reset_index()

idx = nh_offense.groupby(['NeighborhoodName'])['Count'].transform(max) == nh_offense['Count']
nh_offense = nh_offense[idx]

# Maybe a pie chart or a bar chart for this???
common_crimes = nh_offense['UCRCrime'].value_counts().reset_index()

explode = (0.5,0,0,0)

plt.pie(common_crimes['UCRCrime'],explode=explode, labels=common_crimes['index'],
        autopct="%1.1f%%", shadow=False, startangle=140)


# In[26]:


this_df = nh_crime.loc[nh_crime['UCRType']==1]
crime_nh = this_df.groupby(['Neighborhood', 'NeighborhoodName', 'Year'])

# Create scatterplot of total crimes per neighborhood.
nh_tot = crime_nh['Count'].sum().reset_index()

yr_colors = {2008:'red', 2009:'green', 2010:'blue', 2011:'yellow', 2012:'hotpink', 2013:'darkviolet', 2014:'aqua', 2015:'orange'}

x_axis = nh_tot['Year']
x_values = nh_tot['Neighborhood']
y_values = nh_tot['Count']
labels = nh_tot['NeighborhoodName']

plt.scatter(x_values, y_values, marker="o", c=nh_tot['Year'].map(yr_colors), edgecolors="black", alpha=0.5)
            #s=x_values, alpha=0.5)
# plt.xticks(x_values, labels, rotation='vertical')

# Create the regression line to see if there is a relationship between the Neighborhood and the count of crime
(slope, intercept, rvalue, pvalue, stderr) = linregress(x_values, y_values)

regress_values = x_values * slope + intercept

line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))

# plt.scatter(x_values,y_values)
plt.plot(x_values,regress_values,"r-")
plt.annotate(line_eq,(6,10),fontsize=15,color="red")

plt.xlabel('Neighborhoods')
plt.ylabel('Count of Crimes')

plt.title('Crime Trend Across Neighborhoods')

plt.show()


# ## Heat Maps

# In[ ]:





# In[ ]:





# In[ ]:




