
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 30)

# 1. Read dataset

df = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Documents\SKILL 60\CIE 1 MILESTONE\swiggy_file.csv")

original_rows = len(df)
original_duplicates = df.duplicated().sum()

# 2. Study the dataset

print("\n" + "=" * 80)
print("SWIGGY DATASET - FIRST 5 ROWS")
print("=" * 80)
print(df.head().to_string(index=False))

print("\n" + "=" * 80)
print("DATASET SHAPE")
print("=" * 80)
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\n" + "=" * 80)
print("COLUMN NAMES")
print("=" * 80)

column_table = pd.DataFrame({
    "Column Number": range(1, len(df.columns) + 1),
    "Column Name": df.columns
})
print(column_table.to_string(index=False))

print("\n" + "=" * 80)
print("DATA TYPES")
print("=" * 80)

dtype_table = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values
})
print(dtype_table.to_string(index=False))

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

missing_table = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum().values
})
print(missing_table.to_string(index=False))

print("\n" + "=" * 80)
print("DUPLICATE ROWS")
print("=" * 80)
print("Duplicate rows:", original_duplicates)

print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)
print(df.describe(include="all").transpose().to_string())

# 3. Data transformation

# Remove duplicate rows

df = df.drop_duplicates()

# Clean column names

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Convert Rating into numbers

df["rating"] = pd.to_numeric(
    df["rating"].replace(["--", "NEW"], np.nan),
    errors="coerce"
)

# Convert Number of Ratings into numbers

df["number_of_ratings"] = (
    df["number_of_ratings"]
    .astype(str)
    .str.replace(" ratings", "", regex=False)
    .str.replace(" rating", "", regex=False)
    .str.replace("+", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.replace("Too Few Ratings", "", regex=False)
    .str.strip()
)

df["number_of_ratings"] = pd.to_numeric(
    df["number_of_ratings"].str.replace("K", "000", regex=False),
    errors="coerce"
)

# Convert Average Price into numbers

df["average_price"] = (
    df["average_price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(" for two", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["average_price"] = pd.to_numeric(
    df["average_price"], errors="coerce"
)

# Convert Number of Offers into numbers

df["number_of_offers"] = pd.to_numeric(
    df["number_of_offers"], errors="coerce"
)

print("\n" + "=" * 80)
print("DATA AFTER TRANSFORMATION")
print("=" * 80)
print(df.head().to_string(index=False))

print("\nUPDATED DATA TYPES")
print(df.dtypes)

# 4. Handle missing values

print("\n" + "=" * 80)
print("MISSING VALUES BEFORE HANDLING")
print("=" * 80)
print(df.isnull().sum())

# Fill numerical missing values using mean

numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].mean())

# Fill categorical missing values using mode

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])

print("\n" + "=" * 80)
print("MISSING VALUES AFTER HANDLING")
print("=" * 80)
print(df.isnull().sum())

# Display cleaned dataset

print("\n" + "=" * 80)
print("DATASET SHAPE AFTER CLEANING")
print("=" * 80)
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\nDUPLICATES AFTER CLEANING")
print("Duplicate rows:", df.duplicated().sum())

print("\nCLEANED DATASET - FIRST 10 ROWS")
print(df.head(10).to_string(index=False))

# 5. Generate insights

print("\n" + "=" * 80)
print("SWIGGY DATASET INSIGHTS")
print("=" * 80)

# 1. Total restaurants

print("\n1. TOTAL RESTAURANTS")
print("Total restaurant listings:", len(df))

# 2. Total locations

print("\n2. TOTAL LOCATIONS")
print("Number of locations:", df["location"].nunique())

# 3. Top 10 locations

print("\n3. TOP 10 LOCATIONS")

top_locations = df["location"].value_counts().head(10)

location_table = pd.DataFrame({
    "Location": top_locations.index,
    "Restaurant Count": top_locations.values
})
print(location_table.to_string(index=False))

# 4. Top 10 cuisines

print("\n4. TOP 10 CUISINES")

top_cuisines = df["cuisine"].value_counts().head(10)

cuisine_table = pd.DataFrame({
    "Cuisine": top_cuisines.index,
    "Restaurant Count": top_cuisines.values
})
print(cuisine_table.to_string(index=False))

# 5. Pure vegetarian restaurants

print("\n5. PURE VEG RESTAURANTS")

veg_table = df["pure_veg"].value_counts().reset_index()
veg_table.columns = ["Pure Veg", "Restaurant Count"]
print(veg_table.to_string(index=False))

# 6. Rating distribution

print("\n6. RATING DISTRIBUTION")

rating_table = df["rating"].value_counts().head(10).reset_index()
rating_table.columns = ["Rating", "Restaurant Count"]
print(rating_table.to_string(index=False))

# 7. Offer insights

print("\n7. OFFER INSIGHTS")
print("Average number of offers:", round(df["number_of_offers"].mean(), 2))
print("Maximum number of offers:", df["number_of_offers"].max())
print("Total number of offers:", df["number_of_offers"].sum())

# 8. Top 10 restaurant names

print("\n8. TOP 10 RESTAURANTS")

top_restaurants = df["restaurant_name"].value_counts().head(10)

restaurant_table = pd.DataFrame({
    "Restaurant Name": top_restaurants.index,
    "Number of Listings": top_restaurants.values
})
print(restaurant_table.to_string(index=False))

# 9. Average price by location

print("\n9. AVERAGE PRICE BY LOCATION")

price_by_location = (
    df.groupby("location")["average_price"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
print(price_by_location.to_string())

# 10. Cleaning summary

print("\n10. CLEANING SUMMARY")
print("Original rows:", original_rows)
print("Duplicate rows removed:", original_duplicates)
print("Rows after cleaning:", len(df))

# Final insights summary

print("\n" + "=" * 80)
print("FINAL INSIGHTS SUMMARY")
print("=" * 80)

print("Total restaurant listings:", len(df))
print("Different locations:", df["location"].nunique())
print("Most common location:", df["location"].value_counts().index[0])
print("Most common cuisine:", df["cuisine"].value_counts().index[0])
print("Most common Pure Veg category:", df["pure_veg"].value_counts().index[0])
print("Average number of offers:", round(df["number_of_offers"].mean(), 2))
print("Average restaurant rating:", round(df["rating"].mean(), 2))
print("Average price:", round(df["average_price"].mean(), 2))

# 6. Data visualization

# Chart 1: Distribution of restaurant ratings

plt.figure(figsize=(8, 5))
sns.histplot(df["rating"], bins=20, kde=True)
plt.title("Distribution of Restaurant Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig("rating_distribution.png")
plt.show()

# Chart 2: Top 10 locations

plt.figure(figsize=(10, 6))
df["location"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Locations by Restaurant Listings")
plt.xlabel("Location")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("top_locations.png")
plt.show()

# Chart 3: Top 10 cuisines

plt.figure(figsize=(10, 6))
df["cuisine"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Most Common Cuisines")
plt.xlabel("Cuisine")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("top_cuisines.png")
plt.show()

# Chart 4: Average price by location

plt.figure(figsize=(10, 6))
price_by_location.plot(kind="bar")
plt.title("Top 10 Locations by Average Restaurant Price")
plt.xlabel("Location")
plt.ylabel("Average Price")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("average_price_by_location.png")
plt.show()

# Chart 5: Pure Veg restaurant counts

plt.figure(figsize=(7, 5))
df["pure_veg"].value_counts().plot(kind="bar")
plt.title("Pure Veg Restaurant Counts")
plt.xlabel("Pure Veg")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("pure_veg_counts.png")
plt.show()

# 7. Save final cleaned dataset

df.to_csv("swiggy_dataset_cleaned.csv", index=False)

print("\n" + "=" * 80)
print("FINAL DATAFRAME INFORMATION")
print("=" * 80)
df.info()

print("\nFINAL MISSING VALUES")
print(df.isnull().sum())

print("\nCleaned dataset saved as swiggy_dataset_cleaned.csv")
print("All insights and visualizations completed successfully.")
