# 📦 Amazon Review Sentiment Analysis using PyTorch LSTM

A complete, end-to-end Deep Learning Sentiment Analysis system that classifies Amazon customer feedback into **Positive** or **Negative** sentiments. Built using **PyTorch**, **NLTK**, and **Scikit-Learn**, this repository includes the full training workflow, text normalization pipeline, model serialization, and an interactive **Streamlit Web Dashboard** styled with Amazon's dark UI theme.

---

## 📌 Project Overview

- **Full NLP Normalization**: HTML tag removal, non-alphabet filtering, lowercasing, stopword removal, and Porter Stemming.
- **TF-IDF Feature Representation**: Top `5,000` text features extracted using Scikit-Learn's `TfidfVectorizer`.
- **Deep Learning Architecture**: 2-Layer **LSTM** network with Dropout (`0.3`) and Linear classification output.
- **Model Performance**: Achieved **79% Accuracy** across 5,000 unseen test samples.
- **Model Deployment**: Interactive Streamlit web interface featuring real-time inference, confidence scoring, and quick test presets.

---

## 📁 Repository Structure

```text
├── Amazon-Review-Dataset.csv     # Raw Amazon customer review dataset
├── LSTM.ipynb                    # Jupyter notebook for data cleaning, training & evaluation
├── app.py                        # Streamlit dashboard web application
├── tfidf_vectorizer_amazon.joblib# Serialized TF-IDF vectorizer vocabulary
├── rnn_amazon_model.pth          # PyTorch model state dictionary weights
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation

```

---

## 🛠️ Tech Stack & Dependencies

* **Language**: Python 3.x
* **Deep Learning**: PyTorch (`torch.nn`, `torch.optim`)
* **Data & Machine Learning**: `pandas`, `scikit-learn`, `joblib`
* **Natural Language Processing**: NLTK (Tokenizer, Stopwords, PorterStemmer)
* **Web Interface**: Streamlit

---

## 📊 Dataset & Preprocessing Pipeline

1. **Dataset Loading**: Loaded `Amazon-Review-Dataset.csv` and removed missing values via `df.dropna()`.
2. **Sentiment Labeling**:
* Ratings $> 3 \rightarrow$ **1 (Positive)**
* Ratings $\le 3 \rightarrow$ **0 (Negative)**


3. **Text Cleaning Pipeline**:
* Lowercased review strings.
* Stripped HTML tags using regex (`<[^>]+>`).
* Retained alphabetic characters (`[^a-z\s]`).
* Removed standard English stopwords.
* Applied `PorterStemmer` to reduce tokens to root forms.


4. **Vectorization**: Transformed cleaned tokens into dense vectors using `TfidfVectorizer(max_features=5000)`.

---

## 🏗️ Neural Network Architecture

The sequence model uses a 2-layer stacked LSTM architecture with linear projection and BCE loss:

```python
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, num_layers=2, hidden_size=128, dropout_prob=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

```

### Hyperparameters

* **Input Dimension**: `5000` (TF-IDF features)
* **Hidden Layer Units**: `128`
* **LSTM Layers**: `2`
* **Dropout Rate**: `0.3`
* **Loss Function**: `BCEWithLogitsLoss`
* **Optimizer**: `Adam`
* **Epochs**: `10`

---

## 📉 Training Logs

```text
Epoch 1/10  | Loss: 0.3324
Epoch 2/10  | Loss: 0.4420
Epoch 3/10  | Loss: 0.4802
Epoch 4/10  | Loss: 0.1568
Epoch 5/10  | Loss: 0.3459
Epoch 6/10  | Loss: 0.2801
Epoch 7/10  | Loss: 0.2314
Epoch 8/10  | Loss: 0.1078
Epoch 9/10  | Loss: 0.2267
Epoch 10/10 | Loss: 0.2402

```

---

## 📈 Evaluation Results

Evaluated on an independent test split of **5,000 reviews**:

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| **0.0 (Negative)** | 0.82 | 0.84 | **0.83** | 3,000 |
| **1.0 (Positive)** | 0.75 | 0.72 | **0.74** | 2,000 |
| **Accuracy** |  |  | **0.79** | 5,000 |
| **Macro Avg** | 0.79 | 0.78 | 0.78 | 5,000 |
| **Weighted Avg** | 0.79 | 0.79 | 0.79 | 5,000 |

### Confusion Matrix

```text
[[2529   471]
 [ 559  1441]]

```

* **True Negatives**: 2,529
* **False Positives**: 471
* **False Negatives**: 559
* **True Positives**: 1,441

---

## ⚡ How to Run Locally

1. **Clone the Repository**:
```bash
git clone https://github.com/AICatalyst890/Amazon-Review-Sentiment-Analysis.git
cd Amazon-Review-Sentiment-Analysis

```


2. **Install Dependencies**:
```bash
pip install -r requirements.txt

```


3. **Launch the Streamlit Web App**:
```bash
streamlit run app.py

```