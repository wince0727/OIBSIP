import pandas as pd

apps = pd.read_csv("apps.csv")
reviews = pd.read_csv("user_reviews.csv")

print(apps.head())
print(reviews.head())

apps.drop_duplicates(inplace=True)

apps.dropna(subset=['Rating'], inplace=True)

apps['Installs'] = apps['Installs'].astype(str)
apps['Installs'] = apps['Installs'].str.replace('[+,]', '', regex=True)
apps['Installs'] = pd.to_numeric(apps['Installs'], errors='coerce')
apps.dropna(subset=['Installs'], inplace=True)

apps['Price'] = apps['Price'].astype(str).str.replace('$','', regex=False)
apps['Price'] = pd.to_numeric(apps['Price'], errors='coerce').fillna(0)

print("Apps data cleaned")

print("\n Top App Categories:")
print(apps['Category'].value_counts().head(10))

print("\n Average Rating:", round(apps['Rating'].mean(), 2))

top_installed = apps.sort_values(by='Installs', ascending=False).head(10)
print("\n Top Installed Apps:")
print(top_installed[['App','Installs']])

print("\n Price Summary:")
print(apps['Price'].describe())

reviews.dropna(subset=['Translated_Review'], inplace=True)

sentiment_avg = reviews.groupby('App')['Sentiment_Polarity'].mean().reset_index()

print("\n Sentiment sample:")
print(sentiment_avg.head())

import matplotlib.pyplot as plt

apps['Category'].value_counts().head(10).plot(kind='bar')
plt.title("Top App Categories")

plt.savefig("category_chart.png")  
plt.show()


apps['Rating'].plot(kind='hist', bins=10)
plt.title("Rating Distribution")

plt.savefig("rating_chart.png") 
plt.show()


apps.to_csv("cleaned_apps.csv", index=False)
sentiment_avg.to_csv("app_sentiment.csv", index=False)

print("Cleaned files saved")




