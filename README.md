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

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
