# Text Classification on the TTC4900 News Dataset

## Overview
This project explores text classification on the TTC4900 dataset using various natural language processing (NLP) techniques and machine learning models. The study compares traditional machine learning algorithms, advanced neural networks, and state-of-the-art transformer models to identify the best-performing method for Turkish text classification tasks.

---

## Dataset
The **TTC4900 dataset** is a collection of Turkish news articles categorized into 10 distinct classes (e.g., economy, sports, technology). The dataset contains 4900 samples, with an average text length of 120 words.

---

## Methodology
### 1. Preprocessing
- **Tokenization**: Splitting text into individual words.
- **Lemmatization**: Reducing words to their root forms (e.g., "koşuyorum" → "koş").
- **Case Normalization**: Converting text to lowercase.
- **Stopword Removal**: Eliminating common words with minimal semantic value.
- **Special Character Removal**: Removing punctuation and special characters.

### 2. Feature Extraction
- **Bag of Words (BoW)**: Representation based on word frequencies.
- **TF-IDF**: Capturing the importance of terms relative to the document and dataset.
- **Word2Vec**: Using pre-trained embeddings for semantic representation.

### 3. Model Training
#### Traditional Machine Learning Models:
- Random Forest
- XGBoost
- LightGBM
- Artificial Neural Networks (ANN)

#### Advanced Neural Networks:
- Convolutional Neural Networks (CNN)
- Long Short-Term Memory (LSTM)
- Bidirectional LSTM (BiLSTM)
- Gated Recurrent Unit (GRU)

#### Transformer-Based Models:
- **Multilingual BERT**: Pre-trained model for multiple languages, including Turkish.
- **Fine-tuned BERT**: Customized BERT model for the TTC4900 dataset.

### 4. Ensemble Learning
- **Voting Ensemble**: Combining models via cumulative probabilities.
- **Stacking Ensemble**: Training a meta-model to combine predictions.

---

## Performance Evaluation
- **Traditional Models**: Established baseline performances, with TF-IDF generally outperforming BoW.
- **Neural Networks**: Demonstrated enhanced contextual understanding, especially with Word2Vec embeddings.
- **BERT Models**: Achieved the best overall performance, leveraging advanced language understanding capabilities.

---

## Visualizations
- **Confusion Matrix**: Detailed analysis of classification performance.
- **Training and Validation Curves**: Visualizing accuracy and loss for deep learning models.

---

## Key Findings
- Traditional machine learning models provide competitive baselines.
- Advanced neural networks excel in contextual and sequential text understanding.
- BERT models significantly outperform others, highlighting the power of transformer-based approaches for Turkish text classification.

---

## Future Work
- Exploring larger and more diverse datasets.
- Experimenting with additional transformer architectures.
- Developing domain-specific Turkish language models.

---

## References
- TTC4900 Dataset: https://www.kaggle.com/datasets/savasy/ttc4900
- Preprocessing and Feature Engineering Techniques: https://www.kaggle.com/code/alperenclk/for-beginner-nlp-and-word2vec 
- Transformer Models (BERT, Fine-tuning) for dataset: https://huggingface.co/savasy/bert-turkish-text-classification

