import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#load data
@st.cache_data
def load_data():
    health = pd.read_csv(r'heart.csv')
    
    return health

health = load_data()
#Logo
st.image('logo.webp')

# Main title
st.title('Heart Disease Data Insights for Improved Patient Care at Apollo Hospitals by Raja Debnath')

# Subtitle or header
st.header('Exploratory Data Analysis (EDA)')

# Description
st.write('This project analyzes patient data to identify key factors associated with heart disease.')
st.write('The dataset: ')
st.dataframe(health)

# Print column names.
col_name = [col for col in health.columns]
st.write(f'The {len(col_name)} columns are {col_name}')

if st.checkbox('Check the presence of null values:'):
    st.header('Checking for null values:')
    st.write(health.isnull().sum())

if st.checkbox('Check the datatypes of the variables:'):
    st.subheader('Data Types of the Attributes:')
    st.write(health.dtypes)

st.subheader('Continuous numerical variable and categorical variable:')
k1=[]
k2=[]
for col in health.columns:
    if health[col].nunique()>5:
        k1.append(col)
    else:
        k2.append(col)
st.write(f'The Continuous numerical variables are: {k1}\n \n The categorical variables are:{k2}')
st.subheader('Statistical information only for Continuous numerical variables:')
st.write(health[k1].describe())
st.subheader('Unique values for all categorical variables:')
#descrption of categorical attributes.
desc_cat = [
    r'''
    sex: Gender of the patient: 1 = Male, 0 = Female,'
    ''',r'''
    cp(chest pain type): 0 = Asymptomatic (no chest pain); 1 = Typical angina (chest pain related to decrease in blood supply to the heart); 2 = Atypical angina (chest pain not related to the heart); 3 = Non anginal pain (often esophageal spasms or other non cardiac pain),
    ''',r'''
    fbs(fasting blood sugar): 0 = True (fasting blood sugar > 120 mg/dl); 1 = False (fasting blood sugar <= 120 mg/dl),
    ''',r'''
    restecg (resting electrocardiographic results): 0 = Normal; 1 = ST-T wave abnormality; 2 = eft ventricular hypertrophy by Estes' criteria',
    ''',r'''
    exang(exercise induced angina): 1 = Yes; 0 = No,
    ''',r'''
    slope (slope of the peak exercise ST segment): 0 = Upsloping; 1 = Flat; 2 = Downsloping,
    ''',r'''
    ca(number of major vessels colored by fluoroscopy): 0 = No major vessels colored; 1 = One major vessel colored; 2 = Two major vessels colored; 3 = Three major vessels colored,
    ''',r'''
    thal(Thallium stress test result): 0: Not specified; 1: Normal: No significant blood flow issues detected; 2: Fixed defect Permanent lack of blood flow; 3: Reversible defect Temporary lack of blood flow during stress, suggesting ischemia,
    ''',r'''
    target: Presence or absence of heart disease: 1 = Presence of heart disease; 0 = Absence of heart disease
    ''']
desc_num=[
    r'''
    trestbps: Resting blood pressure in mm Hg on admission to the hospital (continuous variable).'
    ''',r'''
    chol: Serum cholesterol in mg/dl (continuous variable).
    ''',r'''
    thalach: Maximum heart rate achieved (continuous variable).
    ''',r'''
    oldpeak: ST depression induced by exercise relative to rest (continuous variable).'
    ''']
if st.checkbox('Details Continuous numerical variables:'):
    st.write(i for i in desc_num)

if st.checkbox('Details and unique values of categorical variables:'):
    unique_dict = {col : health[col].unique() for col in health.columns}
    st.write(f'{key} : {unique_dict[key]}  {desc_cat[i]} \n' for i, key in enumerate(k2))
st.header('`Visualization`')
st.header('Analysis of `target` feature variable')
st.subheader(
    r'''
    Our feature variable of interest is `target`.

    - It refers to the presence of heart disease in the patient.

    - It is integer valued as it contains two integers 0 and 1 - (0 stands for absence of heart disease and 1 for presence of heart disease).
    '''
)

#fig 1
st.subheader('Frequency distribution of \'target\' variable')
# Create the plot
target_labels = {0: 'absence of heart disease', 1: 'presence of heart disease'}
fig, ax = plt.subplots(figsize=(6, 4))
ax = sns.countplot(x=health['target'].map(target_labels), data=health,palette='Set2')
# Add count labels using `bar_label`
for i in ax.containers:
    ax.bar_label(i, fmt='%d', fontsize=12, fontweight='bold')
ax.set_xlabel('heart disease in the patient')
st.pyplot(fig)
st.write(
    r'''
    # Interpretation

       The above plot confirms the findings that -

   - There are 165 patients suffering from heart disease, and 
   
   - There are 138 patients who do not have any heart disease. '''
)

#Groupby target wrt sex
#create dict
st.subheader('Gender-wise Distribution of Heart Disease')
target_labels = {0: 'absence of heart disease', 1: 'presence of heart disease'}
sex_labels = {0 : 'Female', 1 : 'Male'}
# Create mapped columns (without modifying original data)
health['sex_mapped'] = health['sex'].map(sex_labels)
health['target_mapped'] = health['target'].map(target_labels)

# Now group by mapped columns
result = health.groupby('sex_mapped')['target_mapped'].value_counts()
st.write(result)


#fig 2
st.subheader('Frequency distribution of \'target\' variable w.r.t. `sex` variable.')
plt.figure(figsize=(13, 5))

# Ensure order consistency for categorical variables
sex_order = list(sex_labels.values())  # Ensures cp categories appear in the same order
target_order = list(target_labels.values())

# 1st Subplot: Absolute Counts.
plt.subplot(1, 2, 1)
sns.countplot(
    x=health['sex'].map(sex_labels),
    hue=health['target'].map(target_labels),order=sex_order,hue_order=target_order
)
for container in plt.gca().containers:
    plt.gca().bar_label(container, fontsize=12)
plt.title("Heart Disease Count by Sex")

# 2nd Subplot: Percentage of Each Sex.
plt.subplot(1, 2, 2)

# Compute percentages
health_pct = (
    health.groupby('sex')['target']
    .value_counts(normalize=True)  # Convert to percentage
    .mul(100)  # Convert to percentage scale
    .rename_axis(['sex', 'target'])  # Keep correct indexing
    .reset_index(name='percentage')  # Convert to DataFrame
)

# Plot percentage
sns.barplot(x=health_pct['sex'].map(sex_labels),y=health_pct['percentage'],hue=health_pct['target'].map(target_labels),\
           order=sex_order,hue_order=target_order)

# Add labels on bars
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt="%.1f%%", fontsize=12)

plt.title("Heart Disease Percentage by Sex")
plt.ylabel("Percentage (%)")

st.pyplot(plt.gcf())
st.write(
    r'''
    # Interpretation

- The heart disease is more prevalent in female and as per data 75% of total female have heart disease.
- Whereas 45% of total male has heart disease.
- If we combine the total population you will find that male have higher percentage of heart disease. it could be deceptive as male have more numbers than female. 
-  So, out of 96 females - 72 have heart disease and 24 do not have heart disease.
- Similarly, out of 207 males - 93 have heart disease and 114 do not have heart disease. '''
)
#fig 3
sns.catplot(x='sex_mapped',col='target_mapped',data=health,kind='count')
plt.show()
st.pyplot(plt.gcf())
st.write(
    r'''
    # Interpretation
- From this graph we can observe that the number of male patient having heart disease is more but the number of 
male is also very high compare to female. So the percentage of female suffering from heart disease wihtin the 
toal female is more important than total patient. '''
)


# Bivariate Analysis
#creating new variable for attributes which only have digits.
health_1 = health.iloc[:,0:14]
correlation = health_1.corr()

st.header('The target variable is `target`. So, we should check how each attribute correlates with the `target` variable. We can do it as follows:-')
st.write(correlation['target'].sort_values(ascending=False))
st.write(
    r'''
    # Interpretation
- The correlation coefficient ranges from -1 to +1. 

- When it is close to +1, this signifies that there is a strong positive correlation. So, we can see that there is no variable which has strong positive correlation with `target` variable.

- When it is clsoe to -1, it means that there is a strong negative correlation. So, we can see that there is no variable which has strong negative correlation with `target` variable.

- When it is close to 0, it means that there is no correlation. So, there is no correlation between `target` and `fbs`.

- We can see that the `cp` and `thalach` variables are mildly positively correlated with `target` variable. So, I will analyze the interaction between these features and `target` variable.'''
)


#fig 4
st.subheader('Distribution of `cp`')
st.write(desc_cat[1])
plt.figure(figsize=(6,4))
sns.countplot(x='cp',data=health,palette='Set3')
for i in plt.gca().containers:
    plt.gca().bar_label(i)
st.pyplot(plt.gcf())

#fig 5
st.subheader('Frequency distribution of \'target\' variable w.r.t. `cp` variable.')
cp_label = {0 : 'Asymptomatic', 1 : 'Typical angina',2 : 'Atypical angina', 3 : 'Non-anginal'}
# Ensure order consistency for categorical variables
cp_order = list(cp_label.values())  # Ensures cp categories appear in the same order
target_order = list(target_labels.values())

plt.figure(figsize=(14,6))
# For categorical attributes only.
plt.subplot(1,2,1)
sns.countplot(x=health['cp'].map(cp_label),hue=health['target'].map(target_labels),data=health,\
              order=cp_order, hue_order=target_order)
# adding the count number on each bar
for i in plt.gca().containers:
    plt.gca().bar_label(i, fontsize=12)
plt.title("Heart Disease Count by Sex")
plt.xlabel('chest pain type')
# plt.ylim([])
cp_pct = (
    health.groupby('cp')['target'].value_counts(normalize=True).mul(100).rename_axis(['cp','target']).\
    reset_index(name='percentage'))
plt.subplot(1,2,2)
sns.barplot(x=cp_pct['cp'].map(cp_label),y=cp_pct['percentage'],hue=cp_pct['target'].map(target_labels),data=health,\
            order=cp_order, hue_order=target_order)
# adding the count number on each bar
for i in plt.gca().containers:
    plt.gca().bar_label(i,fmt="%.1f%%", fontsize=12)
plt.title("Heart Disease Count by Sex")
plt.xlabel('chest pain type')
plt.ylim([0,120])
st.pyplot(plt.gcf())

st.write(
    r'''
    # Interpretation

- The above plot confirms our below findings,
- People having Typical angina (chest pain related to decrease in blood supply to the heart)(82%), 
    have higher chances of heart disease, following 
    Atypical angina (chest pain not related to the heart)(79.3%),
    Non-anginal pain (often esophageal spasms or other non-cardiac pain)(69.6%).'''
)

st.write(
    r'''
    # Explore `thalach` continuous variable

- `thalach` stands for maximum heart rate achieved.
'''
)
st.write(desc_num[3])
#fig 6
st.subheader('Frequency distribution of `thalach` variable.')
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.distplot(health['thalach'])
plt.subplot(1,2,2)
sns.histplot(health['thalach'])
st.pyplot(plt.gcf())

#fig 7
st.subheader('# Visualize frequency distribution of `thalach` variable wrt `target`')
fig, ax = plt.subplots(1,2,figsize=(12,5))
sns.stripplot(x=health['target'].map(target_labels), y='thalach',data=health,palette='Set2',ax=ax[0])

sns.stripplot(x=health['target'].map(target_labels), y='thalach',data=health,palette='Set2', jitter=0.01,ax=ax[1])

st.pyplot(fig)

#fig 8
st.subheader('# Visualize distribution of `thalach` variable wrt `target` with boxplot')
fig, ax = plt.subplots(figsize=(6,4))
sns.boxplot(x=health['target'].map(target_labels), y='thalach',data=health)
st.pyplot(fig)

st.write(
    r'''
    # Explore `thalach` continuous variable

- The above boxplot confirms our finding that people suffering from heart disease (target = 1) have relatively higher heart rate (thalach) as compared to people who are not suffering from heart disease (target = 0).

- We can see that the `thalach` variable is slightly negatively skewed.

- The people suffering from heart disease (target = 1) have relatively higher heart rate (thalach) as compared to people who are not suffering from heart disease (target = 0).

- People having Typical angina (chest pain related to decrease in blood supply to the heart), have higher chances of heart
    disease, following Atypical angina (chest pain not related to the heart),Non-anginal pain (often esophageal spasms or       other non-cardiac pain)

'''
)

#fig 9
st.subheader('# `exang` and `target`')
st.write(desc_cat[4])
exang_map = {0:"Yes", 1:"No"}
target_order = list(target_labels.values())
plt.figure(figsize=(6,4))
sns.countplot(x=health['exang'].map(exang_map), hue=health['target'].map(target_labels),data=health,hue_order=target_order)
for i in plt.gca().containers:
    plt.gca().bar_label(i, fontsize=12)
plt.xlabel('exercise induced angina')
st.pyplot(plt.gcf())

#fig 10
st.subheader('# `thal` and `target`')
st.write(desc_cat[7])
target_order = list(target_labels.values())
fig, ax = plt.subplots(figsize=(7,5))
sns.countplot(x='thal',hue=health['target'].map(target_labels),data=health,hue_order=target_order,ax=ax)
# for i in plt.gca().containers:
#     plt.gca().bar_label(i, fontsize=12)
ax.set_xlabel('Thallium stress test result')
st.pyplot(fig)


#fig 11
st.subheader('# `fbs` and `target`')
st.write(desc_cat[2])
fbs_label = {1: 'False', 0: 'True'}
target_order = list(target_labels.values())
fig, ax = plt.subplots(figsize=(7,5))
sns.countplot(x=health['fbs'].map(fbs_label), hue=health['target'].map(target_labels), data=health,hue_order=target_order,ax=ax) # For categorical attributes only.
# adding the count number on each bar
# for i in plt.gca().containers:
#     plt.gca().bar_label(i, fontsize = 12)
ax.set_xlabel('fasting blood sugar')
st.pyplot(fig)

st.write(
    r'''
    # Discover patterns and relationships

- I will use `heat map` and `pair plot` to discover the patterns and relationships in the dataset.

- First of all, I will draw a `heat map`.'''
)

#fig 12
fig, ax = plt.subplots(figsize=(14,6))
sns.heatmap(correlation,annot=True)
ax.set_title("Correlation heatmap for Heart Disease Analysis")

st.pyplot(fig)

st.write(
    r'''
# Interpretation

From the above correlation heat map, we can conclude that :-

- `target` and `cp` variable are mildly positively correlated (correlation coefficient = 0.43).

- `target` and `thalach` variable are also mildly positively correlated (correlation coefficient = 0.42).

- `target` and `slope` variable are weakly positively correlated (correlation coefficient = 0.35).

- `target` and `exang` variable are mildly negatively correlated (correlation coefficient = -0.44).

- `target` and `oldpeak` variable are also mildly negatively correlated (correlation coefficient = -0.43).

- `target` and `ca` variable are weakly negatively correlated (correlation coefficient = -0.39).

- `target` and `thal` variable are also waekly negatively correlated (correlation coefficient = -0.34).'''
)

#fig 13

st.subheader('Pair plot')
num_var = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak','target' ]
sns.pairplot(health[num_var], kind='scatter', diag_kind='hist')
st.pyplot(plt.gcf())

st.write(
    r'''
# Comment

- I have defined a variable `num_var`. Here `age`, `trestbps`, ``chol`, `thalach` and `oldpeak`` are numerical variables and `target` is the categorical variable.
- So, I am using this to check relationships between these variables.
- `age` has a negative correlation with `thalach`.
- `age` has a slightly positive correlation with `chol`
-  No other variable has any relation with any other variables.'''
)


#fig 14
st.subheader('Distribution of patient\'s `age`')
st.write(health['age'].describe())
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(health['age'])
plt.subplot(1,2,2)
sns.stripplot(x=health['target'].map(target_labels),y='age',data=health,palette='Set2')
st.pyplot(plt.gcf())

#fig 15
st.subheader('boxplot of patient\'s `age`')
fig, ax = plt.subplots(figsize=(6,4))
sns.boxplot(x=health['target'].map(target_labels),y='age',data=health,ax=ax)
st.pyplot(fig)

st.write(
    r'''
    # Interpretation

- We can see that the people suffering from heart disease (target = 1) and people who are not suffering from heart disease (target = 0) have comparable ages.
- It suggests that the average age, spread, and variability of the ages of both groups do not show a significant difference.
- The mean age of the people who have heart disease is less than the mean age of the people who do not have heart disease.
- The dispersion or spread of age of the people who have heart disease is greater than the dispersion or spread of age of the people who do not have heart disease.'
  '''
)

#fig 16
st.subheader('Jointplot to visualize the relationship between `age` and `thalach` variable')
sns.jointplot(x='age',y='thalach',data=health,hue=health['target'].map(target_labels),hue_order=target_order)
st.pyplot(plt.gcf())

st.write(
    r'''
    #Interpretation

- People having comparetively lower `age` (between 30 to 55) & having high `thalach: Maximum heart rate achieved`, are more prone to heart disease.
'''
)

#fig 17
st.subheader('Outliner detection(boxplot)')

fig, ax = plt.subplots(3,2,figsize=(14,8))
sns.boxplot(x=health['age'],ax=ax[0,0])
ax[0,0].set_title('age')
sns.boxplot(x=health['trestbps'] ,ax=ax[0,1])
ax[0,1].set_title('trestbps')
sns.boxplot(x=health['chol'],ax=ax[1,0])
ax[1,0].set_title('chol')
sns.boxplot(x=health['thalach'],ax=ax[1,1])
ax[1,1].set_title('thalach')
sns.boxplot(x=health['oldpeak'],ax=ax[2,0])
ax[2,0].set_title('oldpeak')

fig.delaxes(ax[2,1])

plt.tight_layout()
st.pyplot(fig)

st.write(
    r'''
    # Interpretation:

- `age` has no outlier

- `trestbps` variable contains outliers to the right side.

- `chol` variable also contains outliers to the right side.

- `thalach` variable contains a single outlier to the left side.

- `oldpeak` variable contains outliers to the right side.

- Those variables containing outliers needs further investigation.
'''
)



# Conclusion Section
st.subheader('Key Insights')
insights = """
- There are 165 patients suffering from heart disease, and 
- There are 138 patients who do not have any heart disease.
- Out of 96 females - 72 have heart disease and 24 do not have heart disease, which is 75%.
- Similarly, out of 207 males - 93 have heart disease and 114 do not have heart disease, which is 45%.
- The heart disease is more prevalent in female and as per data 75% of total female have heart disease.
- Whereas 45% of total male has heart disease.
- People having Typical angina (chest pain related to decrease in blood supply to the heart)(82%), 
    have higher chances of heart disease, following 
    Atypical angina (chest pain not related to the heart)(79.3%),
    Non-anginal pain (often esophageal spasms or other non-cardiac pain)(69.6%).
- people suffering from heart disease have relatively higher heart rate (thalach) as compared to people who are not suffering from heart disease.
- People suffering from heart disease and people who are not suffering from heart disease have comparable ages.
- The mean age of the people who have heart disease is less than the mean age of the people who do not have heart disease.  
- Peoplee having comparetively lower `age` (between 30 to 55) & having high `thalach: Maximum heart rate achieved`, are more prone to heart disease.
"""
st.write(insights)
