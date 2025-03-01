
# Heart Attack Risk Prediction & Obesity Data Analysis

By: Sijian (Cindy) Liang

## Introduction

This project explores two datasets:

1. Heart Attack Risk Dataset: https://www.kaggle.com/datasets/arifmia/heart-attack-risk-dataset

- 50,000 rows, 20 columns covering various demographic, lifestyle, and medical factors influencing heart attack risk.
- Target variable: Heart_Attack_Risk (Low, Moderate, High).
- Features include categorical, numerical, and Boolean attributes related to health metrics.

2. Obesity Prediction Dataset: https://www.kaggle.com/datasets/ruchikakumbhar/obesity-prediction

- 2111 rows, 17 columns providing insights into obesity rates and associated factors such as diet, physical activity, and demographics.
- Used for cross-analysis with the Heart Attack Risk dataset to explore potential relationships between obesity and cardiovascular risk.


By combining these datasets, this project aims to provide a valid data group for exploring patterns, trends, and predictive factors related to cardiovascular diseases and obesity.


## Assignment 3: Heart Attack Risk Prediction Analysis & Visualization 

#### Data Processing & Cleaning
- It is a clean dataset without any null values. 
- Converted binary integer columns into Boolean values for better readability.
- Standardized categorical and numerical features to ensure consistency. (It contains 2 "float64" columns, 7 "object" columns, and 11 "int64" columns. However, 7 out of 11 "int64" columns should be boolean columns. To analyze more conveniently, I converted 7 "int64" columns to "boolean" columns. )


#### Exploratory Data Analysis (EDA) & Visualizations

Heart attack risk may be influenced by various factors, these analyses focus on basic observations and patterns, without delving into causal relationships between specific elements and heart attack risks.


__Data Visualisation:__ Five key analyses were conducted, including:

1. Percentage of Participants in Heart Attack Risk

A breakdown of the prevalence of heart attack risk among 5,000 participants.

2. Gender Distribution of Heart Attack Risks

Analyzing how heart attack risk varies between males and females.

3. Age Group Analysis of High Heart Attack Risk

Insights into the age groups most affected by high heart attack risk.

4. BMI Trends by Age and Gender

Visualization of how BMI changes with age across genders in the high heart attack risk group.

5. Health Metric Summary for High-Risk Participants

A table summarizing key health measures such as cholesterol levels, resting blood pressure, and heart rate for participants at high risk.


## Assignment 5: Merging Obesity Dataset to find a valid data group for further analyses 

__Objective:__ 

- Building upon the work in Assignment 3, merging Obestiy dataset with 'Gender' and 'Age' to assess the realtionship between heart attack risk and obesity.

- By combining these datasets, this project aims to provide a valid data group for exploring patterns, trends, and predictive factors related to cardiovascular diseases and obesity.


#### Data Merging Process

To find the relationship between these two datasets, I followed the below data merging process:

- Import libraries and datasets with updating paths since the file structure on Github has been changed.
- Observing data summaries to ensure merge feasibility
- Find common keys between both datasets and explain the reasons
    - Why Use Age and Gender to Join?
- Convert data types to ensure merging without errors
- Remove duplicates BEFORE merging
    - Aggregate the Heart Attack Dataset to ensure each Age-Gender pair has one unique record
    - Check the rows left after aggregation of Heart Attack Dataset
    - Aggregate the Obesity Dataset to ensure each Age-Gender pair has one unique record
    - Check the rows left after aggregation of the Obesity Dataset
- Merge datasets with common keys and explain the merging way and reasons
- Export valid dataset for researching relationship between heart attack risk and obesity

The detail can be found in "Assignment 5.ipynb". 


## How to Use This Analysis:

- Data analysts can follow the data preprocessing steps for similar medical datasets.
- Healthcare professionals can use the graphs and tables to observe trends in heart disease risks.
- Data analysts can follow the data merging process steps for merging similar datasets, especially remove duplicates by aggregating the data.