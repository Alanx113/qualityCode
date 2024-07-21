import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
Load Dataset I'll use the White wine dataset, since this dataset is bigger than the Red one, idk, I've never drank wine or know what ingredients are in wine, if you have, can you tell me, which Wine have a better taste? In [3]: from google.colab import drive drive.mount('/content/drive') Mounted at /content/drive In [4]: white_wine = pd.read_csv('/content

# Prepare data
X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

EDA
In statistics, exploratory data analysis is an approach of analyzing data sets to summarize their main characteristics, often using statistical graphics and other data visualization methods.

Quality columns data is not properly distributed and so we will convert it into 0, and 1

0 Bad White Wine = 3, 4, 5
1 Good White Wine = 6, 7, 8, 9
wine_data.quality = wine_data.quality.replace([3, 4, 5], 0)
wine_data.quality = wine_data.quality.replace([6, 7, 8, 9], 1)
wine_data.quality.value_counts() # Let's check it
quality
1    855
0    744
Name: count, dtype: int64
okay great!

"""let's make a correlation and construct the heatmap visualization to better understand it."""

wine_data_correlation = wine_data.corr() # correlation

plt.figure(figsize=(18, 10)) # figuring the size

sns.heatmap(
    wine_data_correlation, # data
    annot=True, # annotation
    cmap='summer' # I'll use summer cmap, coz, i like summer
);

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
