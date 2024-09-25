import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style

# Reading the dataset
df = pd.read_csv("AEP_hourly.csv")

# Display basic information about the dataset
print("="*50)
print("First Five Rows ","\n")
print(df.head(2),"\n")

print("="*50)
print("Information About Dataset","\n")
print(df.info(),"\n")

print("="*50)
print("Describe the Dataset ","\n")
print(df.describe(),"\n")

print("="*50)
print("Null Values","\n")
print(df.isnull().sum(),"\n")

# Creating a new dataset with additional date-related columns
dataset = df.copy()
date_format = "%d-%m-%Y %H:%M"
dataset["Month"] = pd.to_datetime(df["Datetime"], format=date_format).dt.month
dataset["Year"] = pd.to_datetime(df["Datetime"], format=date_format).dt.year
dataset["Date"] = pd.to_datetime(df["Datetime"], format=date_format).dt.date
dataset["Time"] = pd.to_datetime(df["Datetime"], format=date_format).dt.time
dataset["Week"] = pd.to_datetime(df["Datetime"], format=date_format).dt.isocalendar().week
dataset["Day"] = pd.to_datetime(df["Datetime"], format=date_format).dt.day_name()
dataset = dataset.set_index("Datetime")
dataset.index = pd.to_datetime(dataset.index, format=date_format)
print(dataset.head(1))

# Plotting the data
fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))

style.use('ggplot')

# Adding a label to the plot element
sns.lineplot(x=dataset["Year"], y=dataset["AEP_MW"], data=df, label='AEP_MW')
sns.set(rc={'figure.figsize':(15,6)})

plt.title("Energy consumption in Year 2004")
plt.xlabel("Date")
plt.ylabel("Energy in MW")
plt.grid(True)
plt.legend()

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)

plt.title("Energy Consumption According to Year")
plt.show()
