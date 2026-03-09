import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

st.title("Email Spam Detection using SVM")

# Load dataset
data = pd.read_csv("spam_dataset.csv", encoding="latin-1")

# Keep only required columns
data = data[['v1', 'v2']]

# Rename columns
data.columns = ['label', 'text']

# Convert labels to numbers
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

# Show dataset preview
st.subheader("Dataset Preview")
st.write(data.head())

# Features and labels
X = data['text']
y = data['label_num']

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train SVM model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

st.subheader("Model Performance")
st.write("Accuracy:", round(acc*100,2), "%")
st.write("Precision:", round(prec*100,2), "%")
st.write("Recall:", round(rec*100,2), "%")

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ax.imshow(cm)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

ax.set_xticks([0,1])
ax.set_yticks([0,1])

ax.set_xticklabels(["Ham","Spam"])
ax.set_yticklabels(["Ham","Spam"])

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center')

st.pyplot(fig)

# Custom Email Test
st.subheader("Test Custom Email")

user_input = st.text_area("Enter Email Text")

if st.button("Predict"):

    if user_input.strip() != "":

        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)

        if prediction[0] == 1:
            st.error("SPAM Email")
        else:
            st.success("HAM Email")

    else:
        st.warning("Please enter some text")