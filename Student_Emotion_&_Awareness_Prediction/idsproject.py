import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    "Student_ID": ["S01", "S02", "S03", "S04", "S05"],
    "Group": ["Treatment", "Control", "Treatment", "Control", "Treatment"],
    "Disorder_Type": ["ADHD", "Autism", "Emotional Disorder", "ADHD", "Autism"],
    "Pre_Self_Awareness": [1, 3, 1, 3, 1],
    "Post_Self_Awareness": [2, 3, 2, 3, 2],
    "Emotion_Tags": ["Frustrated", "Fun, Sad", "Anxious", "Angry", "Happy, Bad"]
}

# Create DataFrame
df = pd.DataFrame(data)

### Classification Task: Predicting Disorder_Type using Emotion_Tags ###

# Vectorize Emotion_Tags (convert text to numerical data)
vectorizer = CountVectorizer()
X_class = vectorizer.fit_transform(df['Emotion_Tags']).toarray()

# Target (Disorder_Type)
y_class = df['Disorder_Type']

# Train-Test Split for Classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train_class, y_train_class)

# Make predictions for Classification
y_pred_class = classifier.predict(X_test_class)

# Evaluate the Classification model
print("### Classification Results ###")
print("Accuracy:", accuracy_score(y_test_class, y_pred_class))
print("\nClassification Report:\n", classification_report(y_test_class, y_pred_class))

# Display Classification Predictions
for i, prediction in enumerate(y_pred_class):
    print(f"Actual: {y_test_class.iloc[i]}, Predicted: {prediction}")

### Regression Task: Predicting Post_Self_Awareness ###

# Pie Chart: Distribution of Disorder Types
df['Disorder_Type'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Disorder Types')
plt.show()

# Encode categorical features for Regression
le = LabelEncoder()
df['Group'] = le.fit_transform(df['Group'])
df['Disorder_Type'] = le.fit_transform(df['Disorder_Type'])

# Features (X) and target (y) for Regression
X_reg = df[['Pre_Self_Awareness', 'Group', 'Disorder_Type']]
y_reg = df['Post_Self_Awareness']

# Train-Test Split for Regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Decision Tree Regressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train_reg, y_train_reg)

# Make predictions for Regression
y_pred_reg = regressor.predict(X_test_reg)

# Display Regression Predictions
print("\n### Regression Results ###")
for i, prediction in enumerate(y_pred_reg):
    print(f"Actual: {y_test_reg.iloc[i]}, Predicted: {prediction}")

# Check if regression prediction is close to actual value
for i, (actual, predicted) in enumerate(zip(y_test_reg, y_pred_reg)):
    if abs(actual - predicted) < 0.5:
        print(f"Prediction {i+1}: Success (Actual: {actual}, Predicted: {predicted})")
    else:
        print(f"Prediction {i+1}: Failure (Actual: {actual}, Predicted: {predicted})")

# Scatter Plot: Actual vs Predicted for Regression
plt.scatter(y_test_reg, y_pred_reg)
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], color='red', linestyle='--')
plt.title('Actual vs Predicted Scores (Regression)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()ṇ