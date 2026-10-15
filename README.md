# SMS Spam Detection — Backtracking, Multi-Model Machine Learning & LLM

## Project Overview

This project is an SMS Spam Detection system that classifies text messages into two categories:

- **HAM:** Normal or legitimate messages.
- **SPAM:** Unwanted, promotional, fraudulent, or suspicious messages.

The project extends a traditional SMS spam classifier by combining text preprocessing, TF-IDF vectorization, handcrafted feature extraction, backtracking-based feature selection, multiple machine learning models, and Large Language Model (LLM) classification.

It also implements a hybrid ML-to-LLM cascade that uses machine learning for confident predictions and an LLM for uncertain cases.

##  Objectives

- Understand and explore SMS data using Basic Exploratory Data Analysis (EDA).
- Clean and preprocess text messages.
- Convert text into numerical representations using TF-IDF.
- Extract useful handcrafted features from SMS messages.
- Select useful features using a backtracking algorithm.
- Train and compare multiple machine learning models.
- Evaluate classification performance using standard metrics.
- Compare classical ML with zero-shot and few-shot LLM classification.
- Build a hybrid system to reduce unnecessary LLM API calls.
- Save the trained model and experimental results for future use.

##  Dataset

**Dataset:** SMS Spam Collection

The project is designed for a dataset containing approximately 5,572 SMS messages, with around 13% spam messages.

The dataset contains two main fields:

| Column | Description |
|---|---|
| `message` | The original SMS text |
| `label` | The message category: HAM or SPAM |

Labels are encoded as:

- `HAM = 0`
- `SPAM = 1`

The notebook supports CSV filenames such as `SMS.csv`, `sms.csv`, and `spam.csv`. If the dataset is not found in the working directory, Google Colab prompts the user to upload a file.

**Note:** The dataset is not included in this repository. Obtain the SMS Spam Collection dataset separately before running the notebook.

## 1. Basic Exploratory Data Analysis (EDA)

EDA is performed to understand the dataset before training the models.

The project includes:

- Checking the dataset shape.
- Counting HAM and SPAM messages.
- Removing missing messages and duplicate message texts.
- Calculating character and word counts.
- Visualizing class distribution.
- Comparing message-length distributions.
- Displaying separate word clouds for HAM and SPAM messages.

These steps help identify class imbalance, common words, and differences between normal and spam messages.

##  2. Text Preprocessing

The project uses NLTK and regular expressions to prepare SMS messages for classification.

Preprocessing steps include:

- Converting text to lowercase.
- Replacing URLs with a common `URL` token.
- Replacing email addresses with an `EMAIL` token.
- Replacing numeric sequences with a `NUMBER` token.
- Removing punctuation.
- Removing English stopwords.
- Applying WordNet lemmatization.

The cleaned text is stored in the `clean_message` column.

##  3. TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical features that machine learning algorithms can process.

The project uses `TfidfVectorizer` with these settings:

- `max_features = 5000`
- `ngram_range = (1, 2)`
- `min_df = 1`
- `sublinear_tf = True`

Using unigrams and bigrams allows the model to learn from individual words and two-word combinations.

For example, phrases such as "free entry" or "claim prize" can provide useful signals for spam detection.

##  4. Handcrafted Feature Extraction

In addition to TF-IDF, the project extracts 12 manually designed features from each original SMS.

| Feature | Purpose |
|---|---|
| `char_len` | Number of characters |
| `word_len` | Number of words |
| `digit_count` | Number of digits |
| `upper_count` | Number of uppercase characters |
| `exclamation_count` | Number of exclamation marks |
| `question_count` | Number of question marks |
| `currency_count` | Number of currency symbols |
| `url_count` | Number of detected URLs |
| `phone_like` | Indicates a phone-number-like pattern |
| `has_free` | Indicates the word "free" |
| `has_win` | Indicates words such as "win", "winner", or "won" |
| `has_call` | Indicates words such as "call" or "contact" |

These features represent common patterns that may appear in spam messages.

##  5. Backtracking Feature Selection

Backtracking is used to search for a useful subset of handcrafted features.

The objective is to maximize the validation Spam F1-score while avoiding unnecessary features.

### How it works

1. Start with an empty feature subset.
2. Choose the next feature.
3. Explore one branch by including the feature.
4. Undo the choice using `selected.pop()`.
5. Explore another branch by excluding the feature.
6. Evaluate each unique subset using a Linear SVM classifier.
7. Keep the best-performing subset.
8. Prefer the smaller subset when scores are tied.

### Optimization techniques

- **Pruning:** Stop extending a branch when the subset reaches six features.
- **Memoization:** Cache scores so each unique subset is evaluated only once.
- **Baseline comparison:** Compare the selected features against a TF-IDF-only model.
- **Search logging:** Record subset sizes, selected features, and validation F1-scores.

With 12 candidate features, there are 4,096 possible subsets. Limiting the subset size to six reduces the number of distinct subsets considered to 2,510.

The search reports the best subset, validation Spam F1-score, baseline score, recursion nodes, and search time.

##  6. Dataset Splitting

The dataset is split into three parts using stratified sampling and a fixed random state.

| Dataset split | Percentage | Purpose |
|---|---:|---|
| Training | 70% | Fit machine learning models |
| Validation | 15% | Evaluate feature subsets and support selection |
| Testing | 15% | Report final model performance |

The random state is set to `42` to make the split reproducible.

The test set is intended for final evaluation rather than feature selection.

##  7. Machine Learning Models

The notebook compares eight classical machine learning approaches:

1. Linear Support Vector Machine (Linear SVM)
2. Logistic Regression
3. Multinomial Naive Bayes
4. Complement Naive Bayes
5. SGD Classifier
6. Random Forest
7. Extra Trees
8. Soft Voting Ensemble (Logistic Regression + Naive Bayes + Random Forest)

The selected handcrafted features are combined with TF-IDF features inside a scikit-learn pipeline.

### Hyperparameter tuning

GridSearchCV with five-fold stratified cross-validation is used to tune three models:

- Linear SVM
- Logistic Regression
- SGD Classifier

The tuned models are also included in the comparison.

##  8. Model Evaluation

The project evaluates classification performance using the following metrics:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted spam messages that are actually spam.
- **Recall:** Proportion of actual spam messages correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures ranking performance across classification thresholds.
- **PR-AUC / Average Precision:** Evaluates the precision-recall trade-off.

The notebook also generates:

- A model comparison table.
- A confusion matrix.
- ROC curves.
- Precision-recall curves.
- A comparison of feature subset performance.

The best model is selected using cross-validation results, and its test performance is reported separately.

##  9. Interpretation and Error Analysis

The project investigates why the classifier makes certain predictions.

### Top spam and HAM indicators

Logistic Regression coefficients are used to identify words or features associated with SPAM and HAM predictions.

### Error analysis

The notebook examines:

- **False Positive (FP):** A normal message incorrectly classified as spam.
- **False Negative (FN):** A spam message incorrectly classified as normal.

Reviewing these errors can reveal weaknesses in the classifier and help guide future improvements.

##  10. SMS Prediction Function

The notebook provides a `predict_sms()` function for classifying new messages using the selected machine learning model.

Example:

```python
predict_sms("Congratulations! You have won a free iPhone.")
predict_sms("Hi, are we meeting tomorrow at 3 PM?")
```

The function returns either `SPAM` or `HAM`.

The notebook also demonstrates predictions on multiple example messages.

##  11. LLM-Based Spam Detection

The project optionally integrates Anthropic's Claude model for text classification.

It compares two LLM strategies:

### Zero-shot classification

The LLM receives classification instructions but no labelled examples.

### Few-shot classification

The LLM receives classification instructions together with labelled HAM and SPAM examples selected from the training data.

The LLM is instructed to return a JSON response containing:

- `label`: SPAM or HAM
- `confidence`: Confidence score
- `reason`: A short explanation

Additional implementation details include:

- API response parsing.
- Parallel API calls.
- Response caching.
- Retry handling.
- Input and output token usage tracking.

The LLM section is optional. An Anthropic API key is required to run it, and API usage may incur charges.

The notebook evaluates LLM approaches on a stratified sample of the test set. The sample size is controlled by `LLM_SAMPLE_SIZE`, which defaults to 300 messages.

##  12. Hybrid ML-to-LLM Cascade

The hybrid system combines classical machine learning with few-shot LLM classification.

It uses a probabilistic ML model to estimate the probability that a message is spam.

The thresholds are:

- If spam probability is at most `0.15`, classify the message using ML as HAM.
- If spam probability is at least `0.85`, classify the message using ML as SPAM.
- If the probability is between `0.15` and `0.85`, use the few-shot LLM prediction.

This approach aims to reduce unnecessary API calls while allowing the LLM to handle uncertain cases.

The notebook compares the following approaches on the same test sample:

- Best classical ML model.
- Zero-shot LLM.
- Few-shot LLM.
- Hybrid ML-to-LLM cascade.

The comparison includes Accuracy, Precision, Recall, Spam F1-score, and the number of LLM calls.

##  13. Optional Free Zero-Shot Transformer

An optional Hugging Face zero-shot classification experiment is included.

It uses the `facebook/bart-large-mnli` model through the Transformers library.

This alternative does not require an Anthropic API key, but it requires downloading a large model and may run slowly without a GPU.

It is disabled by default through:

```python
RUN_HF_ZEROSHOT = False
```

Change the setting to `True` to enable the experiment.

##  14. Saved Output Files

After the relevant notebook cells run successfully, the project saves the following artifacts:

| File | Description |
|---|---|
| `sms_spam_backtracking_model.joblib` | Trained model pipeline, selected features, and model metadata |
| `model_comparison.csv` | Machine learning model comparison results |
| `backtracking_search_log.csv` | Backtracking feature subsets and validation scores |

These files can be downloaded from the Colab working directory and added to a Git repository when appropriate.

**Note:** The notebook's reported metrics are generated during execution. No accuracy or F1-score is assumed in this README.

##  Technologies Used

- Python
- Google Colab / Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- WordCloud
- Scikit-learn
- SciPy
- Joblib
- Anthropic API
- Hugging Face Transformers (optional)
- PyTorch (optional)

##  How to Run the Project

### Option 1: Google Colab

1. Open the `SmS.ipynb` notebook in Google Colab.
2. Upload the notebook to your Google Drive or open it directly in Colab.
3. Upload the SMS dataset when prompted.
4. Run the notebook cells in order.
5. Review the EDA plots and backtracking results.
6. Wait for the machine learning models to train and finish evaluation.
7. Review the model comparison table, confusion matrix, and error analysis.
8. To enable Claude classification, configure `ANTHROPIC_API_KEY` in Colab Secrets.
9. Download the generated CSV files and trained model artifact.

### Option 2: Local Python Environment

Install the required packages:

```bash
pip install nltk wordcloud scikit-learn joblib anthropic tqdm pandas numpy matplotlib seaborn scipy
```

Launch Jupyter Notebook:

```bash
pip install notebook
jupyter notebook
```

Open `SmS.ipynb`, provide the dataset, and run the cells in order.

The notebook uses Google Colab-specific upload and secret-management features, so those sections may need adjustments when running locally.

##  API Key Security

- Never upload your Anthropic API key to GitHub.
- Use Colab Secrets or an environment variable named `ANTHROPIC_API_KEY`.
- Do not commit passwords, tokens, or other private credentials.
- If a key is accidentally published, revoke it and create a replacement.

## Limitations

- Performance depends on the dataset, preprocessing, and model configuration.
- Spam patterns change over time, so new messages may be misclassified.
- The backtracking search evaluates handcrafted feature subsets, not every possible TF-IDF configuration.
- LLM classification requires an external API when using Claude and may incur cost or latency.
- LLM metrics are based on a sample unless the full test set is selected.
- The optional Hugging Face transformer requires additional downloads and computational resources.
- Classification results should be interpreted using multiple metrics rather than accuracy alone.

## Future Improvements

- Test the system on newer SMS and phishing datasets.
- Add a simple web interface for real-time message classification.
- Experiment with additional feature-selection methods.
- Add model calibration and threshold optimization.
- Monitor false positives and false negatives.
- Compare inference speed and API cost across approaches.
- Evaluate the model on a larger independent dataset.

##  Conclusion

This project demonstrates an SMS spam detection workflow that combines text preprocessing, TF-IDF, handcrafted features, backtracking-based feature selection, classical machine learning, model evaluation, and optional LLM integration.

The hybrid cascade explores a practical way to combine fast local ML predictions with LLM-based decisions for uncertain messages.

The notebook generates the evaluation results when executed, allowing the performance of the different approaches to be compared using actual experimental data.

