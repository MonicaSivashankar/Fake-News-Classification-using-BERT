Fake News Detection
Overview
This project aims to develop a model to classify texts as fake news or not. The LIAR dataset is used for training and testing. Various NLP techniques and machine learning models, including transformer-based models, are implemented and evaluated.

Project Structure
Data Collection: Using the LIAR dataset with 12.8k manually labeled phrases from Politifact.
Data Pre-processing: Cleaning text data using NLP techniques and converting it to numerical vectors.
Model Building: Implementing models like Logistic Regression, SVM, Random Forest, XGBoost, and BERT.
Model Evaluation: Comparing models using metrics such as accuracy, precision, recall, AUC, and F1 score.
Dataset
The LIAR dataset consists of short statements labeled by honesty, topic, context, speaker status, party, and date. For binary classification, 'true' and 'mostly-true' are grouped as 'true,' and the rest as 'false'.

Methodology
Data Cleaning: Removing HTML characters, stop-words, punctuations, standardizing words, and removing URLs.
Word Embedding: Using techniques like word count, TF-IDF, and Word2Vec.
Model Development
Traditional ML models: Logistic Regression, SVM, Random Forest, XGBoost.
Transformer-based model: BERT using Hugging Face’s pre-trained models.
Model Evaluation: Using metrics like precision, recall, F1 score, and AUC.
Results
CountVectorizer: Random Forest showed the best performance.
TF-IDF: Improved performance across all models, with Random Forest and XGBoost leading.
BERT: Outperformed traditional models, showing the best F1 score.
Discussion
TF-IDF features generally outperformed CountVectorizer features.
BERT showed superior performance due to its pre-trained architecture, suggesting potential for even better results with larger datasets.
Conclusion
The project compares traditional machine learning models and advanced transformer-based models for fake news detection. TF-IDF features combined with traditional models like Random Forest performed well, while BERT showed the highest accuracy and F1 score.

Future Work
Incorporating more diverse datasets.
Exploring early detection methods to prevent the spread of fake news.
Predicting user tendencies to spread fake news based on historical data.
References
LIAR Dataset
William Yang Wang, "Liar Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection (2017)