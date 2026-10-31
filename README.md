# 🧠 Emotion AI – Text Emotion Detection

An NLP and Machine Learning project that detects emotions from text using **TF-IDF Vectorization** and a **Linear Support Vector Machine (SVM)** classifier.

## 📌 Project Overview

Emotion AI analyzes user-provided text and predicts the emotion expressed in the sentence.

The application uses Natural Language Processing (NLP) techniques to convert text into numerical features and a trained machine learning model to classify emotions.

Users can enter a sentence through an interactive Streamlit web application and instantly receive a predicted emotion.

## ✨ Features

- Text-based emotion detection
- TF-IDF feature extraction
- Linear Support Vector Machine (SVM) classification
- Interactive Streamlit web interface
- Example sentences for testing
- Real-time emotion prediction
- Professional and responsive user interface

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning and NLP |
| TF-IDF | Text feature extraction |
| Linear SVM | Emotion classification |
| Joblib | Saving and loading trained models |
| Streamlit | Web application |

## 📂 Project Structure

```text
Emotion-AI/
│
├── app/
│   └── app.py
│
├── data/
│   └── train.txt
│
├── models/
│   ├── svm_model2.pkl
│   ├── tfidf_vectorizer.pkl
│   └── emotion_labels.pkl
│
├── notebooks/
│   ├── bag_of_words.ipynb
│   └── main.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

### 2. Open the Project Folder

```bash
cd Emotion-AI
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app/app.py
```

The application will open automatically in your browser.

## 🤖 Machine Learning Model

This project uses **Linear Support Vector Machine (Linear SVM)** for emotion classification.

The text is converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)** before being passed to the trained model.

### Model Selection

Three machine learning algorithms were evaluated:

| Algorithm | Bag of Words | TF-IDF |
|---|---:|---:|
| Multinomial Naive Bayes | 76% | 66% |
| Logistic Regression | 84% | 86% |
| Linear SVM | 88% | 89% |

**Selected Model:** Linear SVM with TF-IDF

**Achieved Accuracy:** 89%

*Note: These are the experimental accuracy results obtained during this project's model evaluation.*

## 🖥️ How to Use

1. Launch the Streamlit application.
2. Enter an English sentence in the text input area.
3. Click the **Detect Emotion** button.
4. View the predicted emotion.
5. Try the example sentences to test the application.

## 📦 Saved Model Files

The `models` directory contains:

- `svm_model2.pkl` – Trained Linear SVM classifier.
- `tfidf_vectorizer.pkl` – Fitted TF-IDF vectorizer.
- `emotion_labels.pkl` – Mapping between numeric emotion labels and their names.

All three files are required to run the application.

## 🎯 Project Objective

The objective of this project is to apply Natural Language Processing and Machine Learning techniques to identify emotions from human language and demonstrate the results through an interactive web application.

## 👨‍💻 Author

Developed as a Machine Learning and Natural Language Processing project.

---

**Emotion AI | NLP + Machine Learning + Streamlit**