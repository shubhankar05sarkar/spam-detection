# SMS Spam Detection using NLP

A machine learning project that classifies SMS messages as **Spam** or **Ham (legitimate)** using Natural Language Processing techniques.

---

## Features

- Text preprocessing (cleaning, stopword removal)
- TF-IDF vectorization
- Multinomial Naive Bayes classifier
- Command-line prediction system
- Streamlit web application interface

---

## Technologies Used

- Python
- Scikit-learn
- NLTK
- Pandas
- Streamlit

---

## Dataset

- **SMS Spam Collection Dataset**
- Total messages: **5574**
- Classes:
  - Spam (1)
  - Ham (0)

---

## Methodology

1. Text Preprocessing
   - Lowercasing
   - Removing special characters
   - Stopword removal

2. Feature Extraction
   - TF-IDF Vectorization

3. Model Training
   - Multinomial Naive Bayes

4. Evaluation
   - Accuracy
   - Precision
   - Recall
   - Confusion Matrix

---

## Results

- **Accuracy:** 96.68%
- **Precision (Spam):** 100%
- **Recall (Spam):** 75%

### Confusion Matrix

|                | Predicted Ham | Predicted Spam |
|----------------|--------------|----------------|
| Actual Ham     | 965          | 0              |
| Actual Spam    | 37           | 113            |

---

## Key Insights

- No false positives (no normal message marked as spam)
- Some spam messages are missed (false negatives)
- Model prioritizes safety of legitimate messages

---

## Web Application (Streamlit)

### ✅ Ham Prediction Example
![Ham Screenshot](https://github.com/shubhankar05sarkar/spam-detection/blob/04d82f2971cae2f176713cde636738498a642902/ham.png)

### 🚫 Spam Prediction Example
![Spam Screenshot](https://github.com/shubhankar05sarkar/spam-detection/blob/04d82f2971cae2f176713cde636738498a642902/scam.png)

---
## **Author**

Created with ❤️ by **Shubhankar Sarkar**.  
[GitHub Profile](https://github.com/shubhankar05sarkar)

```bash
git clone https://github.com/your-username/spam-detection-nlp.git
cd spam-detection-nlp
