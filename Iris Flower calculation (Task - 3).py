# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv("C:\\Users\\PC\\OneDrive\\Desktop\\CSB23114\\IRIS.csv")

# Print the first few rows of the dataset to check the structure
print(df.head())

# Check for missing values
print("Missing values in each column:\n", df.isnull().sum())

# Label encoding for species
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Feature columns (sepal length, sepal width, petal length, petal width)
X = df.drop('species', axis=1)  # Features
y = df['species']  # Target

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Print the accuracy score
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Print the classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print the confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualizing the confusion matrix using heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt='g')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Visualizing feature importance
importances = clf.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances in Iris Classification")
plt.show()
