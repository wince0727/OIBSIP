import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sales = pd.read_csv("retail_sales_dataset.csv")
food = pd.read_csv("mcdonalds_nutrition.csv")

print(sales.head())
print(food.head())

sales.drop_duplicates(inplace=True)
sales.dropna(inplace=True)

print(sales.describe())

sales['Date'] = pd.to_datetime(sales['Date'])
sales.groupby('Date')['Total Amount'].sum().plot()

sales['Product Category'].value_counts().head()
food[['Calories', 'Total Fat', 'Sugars']].describe()

sales = pd.read_csv("retail_sales_dataset.csv")
sales['Date'] = pd.to_datetime(sales['Date'])

top_products = sales['Product Category'].value_counts()

top_products.plot(kind='bar')
plt.title("Top Selling Product Categories")
plt.xlabel("Product Category")
plt.ylabel("Number of Sales")
plt.savefig("top_selling_products.png")  
plt.show()


daily_sales = sales.groupby('Date')['Total Amount'].sum()

daily_sales.plot()
plt.title("Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales Amount")
plt.savefig("sales_over_time.png")      
plt.show()


pivot = sales.pivot_table(
    values='Total Amount',
    index='Gender',
    columns='Product Category',
    aggfunc='sum'
)

sns.heatmap(pivot, annot=True, fmt=".0f", cmap="coolwarm")
plt.title("Customer Buying Pattern")
plt.savefig("customer_buying_pattern.png") 
plt.show()

# ---------------- RECOMMENDATIONS ----------------
# 1. Clothing and Electronics are the top-selling product categories.
# 2. Sales show peaks on certain dates, indicating high-demand periods.
# 3. Female customers purchase more Clothing products.
# 4. Male customers spend more on Electronics.
# 5. Businesses should increase stock for top products during peak sales times.



