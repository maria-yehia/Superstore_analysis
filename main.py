import pandas as pd
# 1.1 Importing Data
df = pd.read_csv('superstore_data.csv', encoding='ISO-8859-1')

# 1.2 Exploring Dataset
print(df.head())
print(df.info())
print(df.describe())

# 1.3 Checking for Missing Data
print(df.isnull().sum())

# 1.4 Removing Duplicates
df.drop_duplicates(inplace=True)

# 2. Exploratory Data Analysis
# 2.1 Basic Metrics
print("Basic Statistical Summary:")
print("Total Sales:", df['Sales'].sum())
print("Total Profit:", df['Profit'].sum())

# 2.2 Grouping Data
# By Category
print(df.groupby('Category')['Sales'].sum())
print(df.groupby('Category')['Profit'].sum())

# By Sub-Category
print(df.groupby('Sub-Category')['Sales'].sum())
print(df.groupby('Sub-Category')['Profit'].sum())

# By Region
print(df.groupby('Region')[['Sales', 'Profit']].sum())

# By Segment
print(df.groupby('Segment')[['Sales', 'Profit']].sum())

# Top 10 Customers
print(df.groupby('Customer Name')['Profit'].sum().sort_values(ascending=False).head(10))

# 3. Data Visualization
# 3.1 Bar Charts
import matplotlib.pyplot as plt
import seaborn as sns

# Sales by Category
category_sales = df.groupby('Category')['Sales'].sum().reset_index()
sns.barplot(data=category_sales, x='Category', y='Sales')
plt.title('Sales by Category')
plt.show()

# Profit by Region
region_profit = df.groupby('Region')['Profit'].sum().reset_index()
sns.barplot(data=region_profit, x='Region', y='Profit')
plt.title('Profit by Region')
plt.show()

# 3.2 Scatter Plots
sns.scatterplot(data=df, x='Sales', y='Profit', hue='Category')
plt.title('Sales vs Profit')
plt.show()

# 4. Outliers
# 4.1 Visual Outlier Detection
# Boxplot
sns.boxplot(x=df['Sales']); plt.title("Boxplot of Sales"); plt.show()
sns.boxplot(x=df['Profit']); plt.title("Boxplot of Profit"); plt.show()

# IQR
# Sales Outliers
Q1_sales = df['Sales'].quantile(0.25)
Q3_sales = df['Sales'].quantile(0.75)
IQR_sales = Q3_sales - Q1_sales
outliers_sales = df[(df['Sales'] < (Q1_sales - 1.5 * IQR_sales)) | (df['Sales'] > (Q3_sales + 1.5 * IQR_sales))]
print("Number of Sales Outliers:", outliers_sales.shape[0])

# Profit Outliers
Q1_profit = df['Profit'].quantile(0.25)
Q3_profit = df['Profit'].quantile(0.75)
IQR_profit = Q3_profit - Q1_profit
outliers_profit = df[(df['Profit'] < (Q1_profit - 1.5 * IQR_profit)) | (df['Profit'] > (Q3_profit + 1.5 * IQR_profit))]
print("Number of Profit Outliers:", outliers_profit.shape[0])

# 4.2 Dealing With Outliers
import numpy as np

df['Log_Sales'] = np.log1p(df['Sales'])
df['Log_Profit'] = np.log1p(df['Profit'])

# Plot log-transformed data
sns.boxplot(x=df['Log_Sales'])
plt.title("Boxplot of Log-Transformed Sales")
plt.show()

sns.boxplot(x=df['Log_Profit'])
plt.title("Boxplot of Log-Transformed Profit")
plt.show()

# 5. Checking Assumptions
# 5.1 Normality
from scipy.stats import shapiro
import seaborn as sns
import matplotlib.pyplot as plt

# Visual check with histograms
sns.histplot(df['Log_Sales'], kde=True)
plt.title("Log-Transformed Sales Distribution")
plt.show()

sns.histplot(df['Log_Profit'], kde=True)
plt.title("Log-Transformed Profit Distribution")
plt.show()

# Shapiro-Wilk Normality Test on log-transformed data
log_sales_p = shapiro(df['Log_Sales'])[1]
log_profit_p = shapiro(df['Log_Profit'])[1]

print("Log-Sales Normality p-value:", log_sales_p)
print("Log-Profit Normality p-value:", log_profit_p)

# 5.2 Linearity
sns.regplot(x='Log_Sales', y='Log_Profit', data=df)
plt.title("Log-Transformed Linearity: Sales vs Profit")
plt.show()

# 5.3 Multicollinearity
# Correlation Matrix
# Select only numeric columns
features = ['Log_Sales', 'Log_Profit', 'Quantity', 'Discount']

corr_matrix = df[features].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

features = ['Log_Sales', 'Log_Profit', 'Quantity', 'Discount']

X = df[features].dropna()
X = add_constant(X)

vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)

# 6. Time-Series Analysis
# Ensure Order Date is datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract year and month for grouping
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Year-Month'] = df['Order Date'].dt.to_period('M')  # Monthly period

# Group by month
monthly = df.groupby('Year-Month')[['Sales', 'Profit']].sum().reset_index()
monthly['Year-Month'] = monthly['Year-Month'].astype(str)  # for x-axis

# Plot sales trend
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))
sns.lineplot(data=monthly, x='Year-Month', y='Sales', marker='o')
plt.xticks(rotation=45)
plt.title('📈 Monthly Sales Trend')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()

# Plot profit trend
plt.figure(figsize=(12, 5))
sns.lineplot(data=monthly, x='Year-Month', y='Profit', marker='o', color='green')
plt.xticks(rotation=45)
plt.title('💼 Monthly Profit Trend')
plt.ylabel('Profit')
plt.tight_layout()
plt.show()

# Compare monthly patterns across years
df['Month_Name'] = df['Order Date'].dt.strftime('%b')
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

monthly_by_year = df.groupby(['Year', 'Month_Name'])['Sales'].sum().reset_index()
monthly_by_year['Month_Name'] = pd.Categorical(monthly_by_year['Month_Name'], categories=month_order, ordered=True)
monthly_by_year = monthly_by_year.sort_values(['Year', 'Month_Name'])

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_by_year, x='Month_Name', y='Sales', hue='Year', marker='o')
plt.title('📆 Monthly Sales Comparison by Year')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()
