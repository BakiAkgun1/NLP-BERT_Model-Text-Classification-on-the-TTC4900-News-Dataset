## Overview
This project explores text classification on the TTC4900 dataset using various natural language processing (NLP) techniques and machine learning models. The study compares traditional machine learning algorithms, advanced neural networks, and state-of-the-art transformer models to identify the best-performing method for Turkish text classification tasks.

---

## Dataset
The **TTC4900 dataset** is a collection of Turkish news articles categorized into 10 distinct classes (e.g., economy, sports, technology). The dataset contains:

-**4900 samples** 
-**Number of Classes: 10 (e.g., economy, sports, technology, etc.)**
-**an average text length of 120 words.** 

![image](https://github.com/user-attachments/assets/a7c9198c-7654-406a-af9d-1ab4eff15813)

---

## Methodology
### 1. Preprocessing
- **Tokenization**: Splitting text into individual words.
- **Lemmatization**: Reducing words to their root forms (e.g., "koşuyorum" → "koş").
- **Case Normalization**: Converting text to lowercase.
- **Stopword Removal**: Eliminating common words with minimal semantic value.
- **Special Character Removal**: Removing punctuation and special characters.
- 
![image](https://github.com/user-attachments/assets/bfdbcfe2-5bad-49a7-87dc-1a7796f19a33)


### 2. Feature Extraction
- **Bag of Words (BoW)**: Representation based on word frequencies.
- **TF-IDF**: Capturing the importance of terms relative to the document and dataset.
- **Word2Vec**: Using pre-trained embeddings for semantic representation.

![image](https://github.com/user-attachments/assets/7e18b1d5-43a2-414c-9277-8b9c200e6aa3)

![image](https://github.com/user-attachments/assets/c6896169-e239-4df5-95e3-6e4583811751)

![image](https://github.com/user-attachments/assets/36dee813-48f1-4a74-96c5-9140dc5e2868)

### 3. Model Training
#### Traditional Machine Learning Models:
- Random Forest
- XGBoost
- LightGBM
- Artificial Neural Networks (ANN)
- 
![image](https://github.com/user-attachments/assets/d8df4e57-97e0-48ed-be33-f328446c8b0a)

#### Advanced Neural Networks:
- Convolutional Neural Networks (CNN)
- Long Short-Term Memory (LSTM)
- Bidirectional LSTM (BiLSTM)
- Gated Recurrent Unit (GRU)

#### Transformer-Based Models:
- **Multilingual BERT**: Pre-trained model for multiple languages, including Turkish.
- **Fine-tuned BERT**: Customized BERT model for the TTC4900 dataset.

![image](https://github.com/user-attachments/assets/8e9ecd18-9299-41d1-9166-2b58f874b20a)

### 4. Ensemble Learning
- **Voting Ensemble**: Combining models via cumulative probabilities.
- **Stacking Ensemble**: Training a meta-model to combine predictions.

![image](https://github.com/user-attachments/assets/265d630f-e363-4046-b64b-55ca50cd0321)

---

## Performance Evaluation
- **Traditional Models**: Established baseline performances, with TF-IDF generally outperforming BoW.
- **Neural Networks**: Demonstrated enhanced contextual understanding, especially with Word2Vec embeddings.
- **BERT Models**: Achieved the best overall performance, leveraging advanced language understanding capabilities.

---

## Visualizations
- **Confusion Matrix**: Detailed analysis of classification performance.
- **Training and Validation Curves**: Visualizing accuracy and loss for deep learning models.

![image](https://github.com/user-attachments/assets/109eb259-7362-4d74-b7eb-1336337a1f09)

![image](https://github.com/user-attachments/assets/2de5131f-e981-4cfd-aedd-d5244ac562eb)

![image](https://github.com/user-attachments/assets/9489155d-e90f-4262-ba3e-2c2bbf286110)

### Multilingual BERT

![image](https://github.com/user-attachments/assets/17aa723e-e02a-434b-b115-805c1a47b9e8)

![image](https://github.com/user-attachments/assets/ad2c0b86-ff0a-477c-86bd-64a4b5364883)

### Fine Tuned Bert
![image](https://github.com/user-attachments/assets/2b2841a4-672e-47ca-be54-68de8e59fa4c)

---

## Key Findings
- Traditional machine learning models provide competitive baselines.
- Advanced neural networks excel in contextual and sequential text understanding.
- BERT models significantly outperform others, highlighting the power of transformer-based approaches for Turkish text classification.

![image](https://github.com/user-attachments/assets/f9085ef9-3000-4970-aa67-23ed4a4c9496)

![image](https://github.com/user-attachments/assets/dd4e0a72-3f87-4004-be75-8e7d103b6fc5)

## Findings and Conclusions 
**1. Best Model: Fine-tuned BERT**

![image](https://github.com/user-attachments/assets/ecd64296-a6a1-4d65-8cf7-29b1f82be2cd)

**2. Boosting Algorithms:** XGBoost and LightGBM showed excellent performance in traditional machine learning models. 

![image](https://github.com/user-attachments/assets/15a03c38-ae5d-4530-ba3a-b2e635ee60ce)

**3. Deep Learning Model:** ANN performed excellent performance 

![image](https://github.com/user-attachments/assets/719249bd-6c9f-4a69-840a-5eb8797756cf)

 **4. Transformer Models:** Multilingual BERT and fine-tuned BERT outperformed all other models, proving the effectiveness of transformer-based models in text classification. 
 
![image](https://github.com/user-attachments/assets/dbca0cd5-8be9-4d13-888e-14e80e8d29bc)

---

## References
- TTC4900 Dataset: https://www.kaggle.com/datasets/savasy/ttc4900
- Preprocessing and Feature Engineering Techniques: https://www.kaggle.com/code/alperenclk/for-beginner-nlp-and-word2vec 
- Transformer Models (BERT, Fine-tuning) for dataset: https://huggingface.co/savasy/bert-turkish-text-classification

