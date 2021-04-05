# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:17:53 2021

@author: ramon
"""

# Importing required libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

sns.set_style("white")


# function for the plotting of the biomarkers
def biomarker_plot(data, biomarker):
    """ Create a violinplot of the distributions of the different biomarkers
    inputs the dataframe and the biomarker as a string """
      
    ax = sns.violinplot(x="diagnosis", y="creatinine", data=data)
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("ng/mg Creatinine")
    ax.set_title(f"Levels of {biomarker}")
    plt.show()
    return ax
    

# Load the dataset
data = pd.read_csv("data/pancreatic_data.csv")
print(data.head())

## Distribution of variables
## Age distribution
# Check for NaN values
data["age"].isnull().values.any() #No NaN values

# Histogram age
sns.set_style("white") #Sets the theme of the histogram
sns.distplot(data["age"], kde=False, bins=100)
plt.title("Age distribution", fontsize=16)
plt.xlabel("Age (years)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)

plt.tight_layout()
plt.show

## Sex distribution
# Check for NaN values
data["sex"].isnull().values.any() #No NaN values

# Pie chart sex
sizes_sex = data["sex"].value_counts()
labels_sex = ["F", "M"]
explode_sex = [0, 0.1]
colors = ["#ff9999","#66b3ff"]

sex_fig1, sex_ax1 = plt.subplots()
sex_ax1.pie(sizes_sex, explode=explode_sex, labels=labels_sex, colors=colors,
        autopct="%1.1f%%", shadow=True, startangle=90)
sex_ax1.axis("equal")
plt.tight_layout()
plt.show


## Diagnoses distribution
# Check for nan values
data["diagnosis"].isnull().values.any() # No nan values

# Pie chart diagnosis
sizes_diagnosis = data["diagnosis"].value_counts()
labels_diagnosis = ["Benign", "PDAC", "Healthy"]
explode_diagnosis = [0, 0.1, 0]
colors_diagnosis = ["#ff9999","#66b3ff","#99ff99"]

diagnosis_fig1, diagnosis_ax1 = plt.subplots()
diagnosis_ax1.pie(sizes_diagnosis, explode=explode_diagnosis, 
                  labels=labels_diagnosis, colors=colors_diagnosis,
                  autopct="%.1f%%", shadow=True, startangle=90)
diagnosis_ax1.axis("equal")

plt.tight_layout()
plt.show

# Diagnosis distribution vs age with sex difference
# Change names of diagnosis
data.diagnosis[data.diagnosis == 1] = "Control"
data.diagnosis[data.diagnosis == 2] = "Benign"
data.diagnosis[data.diagnosis == 3] = "PDAC"
sns.catplot(x="diagnosis", y="age", hue="sex", kind="swarm", data=data)
plt.show()

## Distribution of the biomarkers, normalise for creatine
 
## LYVE1
# Normalize LYVE1 to creatinine
data_LYVE1 = data[["diagnosis", "creatinine", "LYVE1"]]
data_LYVE1["creatinine"] = data_LYVE1["LYVE1"]/data_LYVE1["creatinine"]

# Create violinplot of diagnosis vs creatinine 
# to display the distribution of the biomarker
biomarker_plot(data_LYVE1, "LYVE1")

## REG1B
# Normalize REG1B to creatinine
data_REG1B = data[["diagnosis", "creatinine", "REG1B"]]
data_REG1B["creatinine"] = data_REG1B["REG1B"]/data_REG1B["creatinine"]

# Create violinplot of diagnosis vs creatinine 
# to display the distribution of the biomarker
biomarker_plot(data_REG1B, "REG1B")

## TFF1
# Normalize TFF1 to creatinine
data_TFF1 = data[["diagnosis", "creatinine", "TFF1"]]
data_TFF1["creatinine"] = data_TFF1["TFF1"]/data_TFF1["creatinine"]

# Create violinplot of diagnosis vs creatinine 
# to display the distribution of the biomarker
biomarker_plot(data_TFF1, "TFF1")
