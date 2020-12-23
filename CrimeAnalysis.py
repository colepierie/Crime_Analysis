#!/usr/bin/env python
# coding: utf-8

# # Crime Data from St. Louis Missouri (2008-2015)
# ## Loading and Cleaning the Data

# In[2]:


import gmaps
import numpy as np
import pandas as pd
import requests
import time
import matplotlib.pyplot as plt
from scipy.stats import linregress
import scipy.stats as st
from config import g_key
import gmaps
import os


# In[3]:


# Prepare list of files to load.
load_urls = ['2008_data.csv', '2009.csv', '2010.csv', '2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv']

# Create list of columns we want to use in the final DF.
column_names = ['CADAddress', 'CADStreet', 'CodedMonth', 'Count', 'Crime', 'ShortCrimeCode', 'UCRType',                'UCRCrime', 'DateOccured', 'Description', 'District', 'FlagCrime', 'FlagUnfounded',                'ILEADSAddress','ILEADSStreet', 'LocationComment', 'LocationName', 'Neighborhood',                'NeighborhoodName', 'NeighborhoodPrimaryDistrict', 'NeighborhoodAddlDistrict', 'Latitude',                'Longitude', 'Year']


# In[4]:


# Create empty DF for crime data
crime_df = pd.DataFrame(columns = column_names)

# Read single year crime files and store into final data frame
for file in load_urls:
    load_df = pd.read_csv(f'Resources/{file}')
    
    # Select only the columns we want from the single year csv.
    load_df = load_df[column_names]
    
    # Append data from the single year csv into the final DF
    crime_df = crime_df.append(load_df,ignore_index=True)


# In[5]:


# Preview data
crime_df.head()


# In[6]:


# How many records do we have?
crime_df['CADAddress'].count()


# In[7]:


# Remove rows that do not have location information complete.
crime_df = crime_df.loc[(pd.isna(crime_df['Latitude'])==False) & (pd.isna(crime_df['Longitude'])==False) & (pd.isna(crime_df['NeighborhoodName'])==False)]


# In[8]:


# How many records do we have after dropping nulls?
crime_df['CADAddress'].count()


# In[9]:


# I had issues with indexing into the list created by the split string funtion for reach row so I created a new column for the
## split string then did a for loop to assign the year from that column to the year column if the year column is null.

crime_df['Year_calc'] = crime_df['CodedMonth'].str.split('-')

for index in crime_df.index:
    if pd.isna(crime_df['Year'][index]) == True:
        crime_df['Year'][index] =  crime_df['Year_calc'][index][0]
crime_df.head()


# In[10]:


# Remove the no longer needed Year_calc column
crime_df.drop('Year_calc', inplace=True, axis=1)


# In[11]:


# Change the data type of the year column to be an int to remove the decimal
crime_df['Year'] = crime_df['Year'].astype('int')

crime_df.head()


# In[12]:


# Remove row where CADStreet is unknown
crime_df = crime_df.loc[crime_df['CADStreet'] != 'unknown 0000']


# In[13]:


# Check if we can map the Nieghborhood column to a value from another column from the same street, if null.

neighborhoods = crime_df.groupby(['CADStreet','NeighborhoodName']).count()
neighborhoods

## Looks like CADStreet and Neighboborhood do not have a 1-1 relationship, as suspected. So we should not try to map this.
### When we do analysis on Neighborhoods we will just need to keep that in mind. 


# In[14]:


crime_df


# ## Trends Per Year

# In[15]:


#grouping by type of crime and year
yoc_df = crime_df.groupby(['UCRCrime', 'Year']).count()
yoc_df


# In[16]:


# Seperate Drug Abuse Violations
drug_crime = yoc_df.loc["Drug Abuse Violations"].reset_index()
drug_crime


# In[18]:


# Make a line chart to see the amount of arrests per year
drug_crime.plot.line(x="Year", y="Count")

plt.title("Drug Arrests")
plt.xlabel("Years")
plt.ylabel("Number of Arrests")

# Save the graph
plt.savefig('Output/DrugArrests_Yr.png')

# Show the graph
plt.show()


# In[19]:


# pull out the different types of drug arrests 

drug_crime_heroin = crime_df.loc[(crime_df["Description"]== "DRUGS-POSSESSION/HEROIN") &                                  (crime_df["UCRCrime"]== "Drug Abuse Violations")]

heroin_groupby = drug_crime_heroin.groupby(['Year']).count()
heroin_groupby = heroin_groupby.reset_index()
heroin_groupby


# In[20]:


# Get number of Marijuana arrests by year
drug_crime_pot = crime_df.loc[(crime_df["Description"]== "DRUGS-POSSESSION/MARIJUANA") &                                  (crime_df["UCRCrime"]== "Drug Abuse Violations")]
pot_groupby = drug_crime_pot.groupby(['Year']).count()
pot_groupby = pot_groupby.reset_index()
pot_groupby


# In[21]:


# Get number of cocaine arrests by year
drug_crime_cocaine = crime_df.loc[(crime_df["Description"]== "DRUGS-POSSESSION/COCAINE") &                                  (crime_df["UCRCrime"]== "Drug Abuse Violations")]
cocaine_groupby = drug_crime_cocaine.groupby(['Year']).count()
cocaine_groupby = cocaine_groupby.reset_index()
cocaine_groupby


# In[23]:


# Plot the number of arrests for marijuana
pot_plot, = plt.plot(pot_groupby["Year"], pot_groupby["Count"], color="green", label="MARIJUANA" )

# Plot the number of arrests for heroin
herion_plot, = plt.plot(heroin_groupby["Year"], heroin_groupby["Count"], color="red", label="HEROIN" )

# plot the number of arrests for cocaine
cocaine_plot, = plt.plot(cocaine_groupby["Year"], cocaine_groupby["Count"], color="blue", label="COCAINE" )

# Create a legend for our chart
plt.legend(handles=[pot_plot, herion_plot, cocaine_plot], loc="best")

# Add lables
plt.title("Drug Arrests by Drug")
plt.xlabel("Years")
plt.ylabel("Number of Arrests")

# Save the graph
plt.savefig('Output/DrugArrests_Yr_DrugType.png')

# Show the chart
plt.show()


# In[24]:


#check for 1 vs 2 offenses
violent = crime_df.groupby(['UCRType']).count()["Count"]
violent


# In[26]:


# Make a pie Chart of type1 v 2
v_nv_pie = violent.plot(kind="pie", y='Count', title="URCType 1 Vs. URCType 2", autopct="%2.2f%%")
v_nv_pie.set_ylabel("Count")

# Save chart
plt.savefig('Output/PctSevereCrime_Tot.png')

# Show chart
plt.show()


# ## Offenses Per Neighborhood

# In[27]:


# Remove where the Neighborhood is unknown.
nh_crime = crime_df.loc[crime_df['NeighborhoodName']!= 'Unknown']

# Create a DF that is a simple groupby NeighborhoodName to reference in the overall analysis.
crime_nh = nh_crime.groupby(['Neighborhood', 'NeighborhoodName'])

# Create DF that holds the total count of crimes per neighborhood.
nh_tot = crime_nh['Count'].sum().reset_index()

# Take only the largest 20 crime counts (89 neighborhoods makes the graph impossible to read.)
nh_tot_top = nh_tot.nlargest(20,'Count')


# In[28]:


# Plot as a bar chart.
plt.bar(nh_tot_top['NeighborhoodName'], nh_tot_top['Count'], color="b", align="center")
plt.xticks(nh_tot_top['NeighborhoodName'], nh_tot_top['NeighborhoodName'], rotation='vertical')

plt.xlabel('Neighborhood')
plt.ylabel('Total Crime Counts')

plt.title('Top 20 Neighborhoods with Highest Total Recorded Crime')

plt.savefig('Output/WorstNeighborhoods.png')

plt.show()


# In[29]:


# For the top 20 most "dangerous" Neighborhoods, rank by percent of violent crimes.
top_twenty = pd.merge(crime_df,nh_tot_top, on=['NeighborhoodName', 'Neighborhood'], how='right',suffixes=('', '_Total') )
tp_twenty = top_twenty.groupby(['NeighborhoodName', 'UCRType']).agg({'Count':'sum', 'Count_Total':'mean'}).reset_index()

tp_twenty['Pct'] = tp_twenty['Count']/top_twenty['Count_Total']
# tp_twenty

# Attempt to plot top 20 with severe/non-severe crimes seperated
colors = {1:'red', 2:'blue'}
c = tp_twenty['UCRType'].apply(lambda x: colors[x])

# Plot as a bar chart.
plt.bar(tp_twenty['NeighborhoodName'], tp_twenty['Count'], color=c, align="center")
plt.xticks(tp_twenty['NeighborhoodName'], tp_twenty['NeighborhoodName'], rotation='vertical')

plt.xlabel('Neighborhood')
plt.ylabel('Total Crime Counts')

plt.title('Top 10 Neighborhoods with Highest Total Recorded Crime')

plt.savefig('Output/WorstNeighborhoods_UCRPct.png')

plt.show()


# In[30]:


# Create DF with all data from top 20 nieghborhoods
tp_twenty_pct = top_twenty.groupby(['UCRType']).agg({'Count':'sum'}).reset_index()
# top_twenty

# Plot top 20 pct of severe crime
explode = (0.1,0)

labels={1:'Severe', 2:'Non-Severe'}
l = tp_twenty_pct['UCRType'].apply(lambda x: labels[x])

plt.title('Top 20 Neighborhoods with Highest Total Recorded Crime')
plt.pie(tp_twenty_pct['Count'],explode=explode, labels=l,
        autopct="%1.1f%%", shadow=False, startangle=140)


# In[40]:


# Show counts
tp_twenty_pct


# In[33]:


# Graph the overall count of crimes per year per neighborhood in the top 20
tp_year = top_twenty.groupby(['NeighborhoodName', 'Year'])['Count'].sum().reset_index()

plt.plot(tp_year['Year'], tp_year['Count'], label="Crime Tends across Neighborhoods")


# In[34]:


# Graph the overall count of crimes per year in the top 20
tp_year_tot = top_twenty.groupby(['Year'])['Count'].sum().reset_index()

plt.title('Total Crime Reports per Year (Upper 20 Neighborhoods)')
plt.xlabel('Year')
plt.ylabel('Total Crime Counts')

plt.plot(tp_year_tot['Year'], tp_year_tot['Count'], label="Crime Tends across Neighborhoods")

plt.savefig('Output/WorstNeighborhoods_YrChg.png')

plt.show()


# In[35]:


# Take only the smallest 20 crime counts (89 neighborhoods makes the graph impossible to read.)
nh_tot_bottom = nh_tot.nsmallest(20,'Count')

# Plot as a bar chart.
plt.bar(nh_tot_bottom['NeighborhoodName'], nh_tot_bottom['Count'], color="b", align="center")
plt.xticks(nh_tot_bottom['NeighborhoodName'], nh_tot_bottom['NeighborhoodName'], rotation='vertical')

plt.xlabel('Neighborhood')
plt.ylabel('Total Crime Counts')
plt.title('Top 20 Neighborhoods with Lowest Total Recorded Crime')

plt.savefig('Output/BestNeighborhoods.png')

plt.show()


# In[36]:


# For the top 20 least "dangerous" Neighborhoods, graph pct of violent crimes.
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


# In[37]:


# Show overall pct of severe crimes in the bottom 20
bottom_twenty_pct = btm_twenty.groupby(['UCRType']).agg({'Count':'sum'}).reset_index()

explode = (0.1,0)

labels={1:'Severe', 2:'Non-Severe'}
l = bottom_twenty_pct['UCRType'].apply(lambda x: labels[x])

plt.pie(bottom_twenty_pct['Count'],explode=explode, labels=l,
        autopct="%1.1f%%", shadow=False, startangle=140)
plt.title('Top 20 Neighborhoods with Lowest Total Recorded Crime')

plt.savefig('Output/WorstNeighborhoods_UCRPie.png')


# In[41]:


# Show counts
bottom_twenty_pct


# In[42]:


# Graph the overall count of crimes per year per neighborhood in the bottom 20
btm_year = bottom_twenty.groupby(['NeighborhoodName', 'Year'])['Count'].sum().reset_index()

plt.plot(btm_year['Year'], btm_year['Count'], label="Crime Tends across Neighborhoods")


# In[43]:


# Graph the overall count of crimes per year in the bottom 20
btm_year_tot = bottom_twenty.groupby(['Year'])['Count'].sum().reset_index()

plt.title('Total Crime Reports per Year (Lower 20 Neighborhoods)')
plt.xlabel('Year')
plt.ylabel('Total Crime Counts')

plt.plot(btm_year_tot['Year'], btm_year_tot['Count'], label="Crime Tends across Neighborhoods")

plt.savefig('Output/BestNeighborhoods_YrChg.png')

plt.show()


# In[44]:


# Show overall most common crimes per neighborhood
crime_nh = nh_crime.groupby(['NeighborhoodName','UCRCrime'])
nh_offense = crime_nh['Count'].sum().reset_index()

idx = nh_offense.groupby(['NeighborhoodName'])['Count'].transform(max) == nh_offense['Count']
nh_offense = nh_offense[idx]

# Maybe a pie chart or a bar chart for this???
common_crimes = nh_offense['UCRCrime'].value_counts().reset_index()

explode = (0.5,0,0,0)

plt.pie(common_crimes['UCRCrime'],explode=explode, labels=common_crimes['index'],
        autopct="%1.1f%%", shadow=False, startangle=140)


# In[51]:


# Try doing a regression analysis to see if there is a relationship between nieghborhood and number of crimes reported.
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

# Save Image
plt.savefig('Output/Neighborhood_CrimeReports_Yr.png')

plt.show()


# ## Heat Maps

# ### Attempt at using heatmaps. Ultimately could not due to the size of the data set.

# In[46]:


fifteen = pd.read_csv('Resources/2015.csv')  
fifteen = fifteen.loc[(pd.isna(fifteen['Latitude'])==False) 
                        & (pd.isna(fifteen['Longitude'])==False) 
                        & (pd.isna(fifteen['NeighborhoodName'])==False)]

fifteen['Year_calc'] = fifteen['CodedMonth'].str.split('-')

for index in fifteen.index:
    if pd.isna(fifteen['Year'][index]) == True:
        fifteen['Year'][index] =  fifteen['Year_calc'][index][0]
        
fifteen.drop('Year_calc', inplace=True, axis=1)

one_five = fifteen[["UCRType","UCRCrime","Count","Latitude","Longitude"]]


# In[47]:


#Create data sets for Larceny type of crime 
larce = one_five.loc[one_five['UCRCrime'] == 'Larceny-theft']
larce = larce.loc[larce["Count"] >= 1]
#Create locations
lat_lng_lar = larce[["Latitude","Longitude"]]
#Create Weight
weight = larce["Count"]

#Create data sets for Aggravated Assault type of crime 
a_a = one_five.loc[one_five['UCRCrime'] == 'Aggravated Assault']
a_a = a_a.loc[a_a["Count"] >= 1]
#Create locations
lat_lng_a = a_a[["Latitude","Longitude"]]
#Create Weight
weight_a = a_a["Count"]


# In[48]:


#Use fig and layer to map on data set 
fig_one = gmaps.figure()

heat_layer_one = gmaps.heatmap_layer(lat_lng_lar, weights=weight, point_radius = 5)

fig_one.add_layer(heat_layer_one)

#Larceny in STL in year 2015
fig_one


# In[49]:


fig_two = gmaps.figure()

heat_layer_two = gmaps.heatmap_layer(lat_lng_a, weights=weight_a, point_radius = 5)

fig_two.add_layer(heat_layer_two)

fig_two


# ## Additional Analysis

# In[50]:


minicrime = crime_df[["Count", "Year"]]
overall_crime = minicrime.groupby(['Year']).sum()


# In[52]:


tot_crime = overall_crime.plot(kind="line", title="Total # of Crimes Reported From 2008-15 in STL")
tot_crime.set_xlabel("Year")
tot_crime.set_ylabel("Sum of Crimes")

# Save Graph
plt.savefig('Output/TotalCrimeTrend.png')

# Show Graph
plt.show()


# In[53]:


years = crime_df[["Count", "Year", 'UCRType']]

lev_two = years.loc[years["UCRType"] >= 2]
lev_two = lev_two.drop(columns=['UCRType'])
lev_two_group = lev_two.groupby(['Year']).sum()

lev_one = years.loc[years["UCRType"] <= 1]
lev_one = lev_one.drop(columns=['UCRType'])
lev_one_group = lev_one.groupby(['Year']).sum()


# In[54]:


# Plot the Level One Crimes as a line chart
level_one_crimes = plt.plot(lev_one_group, color="blue", label='Type One')

# Plot the Level Two Crimes as a line chart
level_two_crimes = plt.plot(lev_two_group, color="red", label='Type Two')

# Create a legend for our chart along with labels
plt.legend()
plt.title("Total # of Level 1 and 2 Crimes Reported From 2008-15 in STL")
plt.xlabel("Years")
plt.ylabel("Sum of Crimes")

# Save Graph
plt.savefig('Output/TotalCrimeTrend_Severity.png')

# Show Graph
plt.show()


# In[ ]:




