
from sklearn.feature_extraction import DictVectorizer
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


dev_file = 'UP-1.0/UP_English-EWT/en_ewt-up-dev.conllu'
test_file = 'UP-1.0/UP_English-EWT/en_ewt-up-test.conllu'
train_file = 'UP-1.0/UP_English-EWT/en_ewt-up-train.conllu'


with open(dev_file, 'r') as f:
    dev_data = f.readlines()

with open(test_file, 'r') as f:
    test_data = f.readlines()

with open(train_file, 'r') as f:
    train_data = f.readlines()

print("First 20 lines of the dev data:")
for line in dev_data[:30]:
    print(line.strip())

def parse_conllu(conllu_file):
    data = []
    sentence = []

    with open(conllu_file, 'r') as f:
        for line in f:
            # Ignore comment lines
            if line.startswith('#'):
                continue

            # Process token lines
            columns = line.strip().split('\t')
            if len(columns) == 10:
                token = {
                    'ID': columns[0],
                    'FORM': columns[1],
                    'LEMMA': columns[2],
                    'UPOS': columns[3],
                    'XPOS': columns[4],
                    'FEATS': columns[5],
                    'HEAD': columns[6],
                    'DEPREL': columns[7],
                    'UPRED': columns[8],
                    'UPARGS': columns[9]
                }
                sentence.append(token)
            elif len(line.strip()) == 0:  # End of sentence
                if sentence:
                    data.append(sentence)
                    sentence = []
    return data

# Parse development data
dev_parsed = parse_conllu(dev_file)

# Preview the parsed data
print(f"Parsed Dev Data (First sentence): {dev_parsed[0]}")
print(f"Total Dev Sentences: {len(dev_parsed)}")

# Function to replicate sentences for each predicate
def replicate_sentence_for_predicates(sentence):
    replicated_sentences = []

    # Identify the predicates (those with 'UPRED' field filled)
    predicates = [i for i, token in enumerate(sentence) if token['UPRED'] != '_']

    for predicate_idx in predicates:
        new_sentence = []

        # For each token, get its features and label based on its relationship to the predicate
        for token in sentence:
            features = {
                'word': token['FORM'],
                'lemma': token['LEMMA'],
                'pos': token['UPOS'],
                'head': token['HEAD'],
                'deprel': token['DEPREL'],
                'predicate_lemma': sentence[predicate_idx]['LEMMA'],
                'distance_to_predicate': abs(int(token['ID'].split('.')[0]) - int(sentence[predicate_idx]['ID'].split('.')[0]))
            }

            # Assign semantic role label based on the predicate
            role_label = 'O'  # Default: Outside any argument role

            # Improved role extraction logic using UPARGS
            if token['UPARGS'] != '_':
                # Extract role from UPARGS column (e.g., Arg1, ArgM-TMP)
                role_label = token['UPARGS'].split(':')[0]

            # Add token with features and label
            new_sentence.append({'features': features, 'label': role_label})

        # Add the new sentence created for the current predicate to the list
        replicated_sentences.append(new_sentence)

    print(f"Predicates in sentence: {predicates}")
    print(f"Number of replicated sentences: {len(replicated_sentences)}")
    return replicated_sentences

# Parse the data for training, testing, and validation sets
train_parsed = parse_conllu(train_file)
dev_parsed = parse_conllu(dev_file)
test_parsed = parse_conllu(test_file)

# Replicate the sentences for each predicate and append correctly
processed_data_train = []
processed_data_dev = []
processed_data_test = []

# Replicate the sentences for each predicate
for sentence in train_parsed:
    processed_data_train.extend(replicate_sentence_for_predicates(sentence))

for sentence in dev_parsed:
    processed_data_dev.extend(replicate_sentence_for_predicates(sentence))

for sentence in test_parsed:
    processed_data_test.extend(replicate_sentence_for_predicates(sentence))

# Check the size of data before and after preprocessing
num_tokens_before = sum(len(sentence) for sentence in train_parsed + dev_parsed + test_parsed)
num_sentences_before = len(train_parsed + dev_parsed + test_parsed)

num_tokens_after = sum(len(sentence) for sentence in processed_data_train + processed_data_dev + processed_data_test)
num_sentences_after = len(processed_data_train + processed_data_dev + processed_data_test)

print(f"Tokens before preprocessing: {num_tokens_before}")
print(f"Sentences before preprocessing: {num_sentences_before}")
print(f"Tokens after preprocessing: {num_tokens_after}")
print(f"Sentences after preprocessing: {num_sentences_after}")



# Function to extract exactly three features and combine them, using One-Hot Encoding
def extract_features(sentence, predicate_idx, vectorizer=None):
    extracted_data = []
    features_list = []

    # Debug: Print the structure of the first token to inspect the keys available
    print(f"First token in sentence: {sentence[0]}")

    # Debug: Print the entire sentence structure
    print(f"Processing sentence: {[token for token in sentence]}")

    # Check if predicate_idx is valid
    if predicate_idx < 0 or predicate_idx >= len(sentence):
        print(f"Error: Invalid predicate_idx {predicate_idx} for sentence of length {len(sentence)}")
        return [], vectorizer  # Return empty if index is out of range

    # Retrieve the predicate information
    predicate_token = sentence[predicate_idx]
    print(f"Predicate token at index {predicate_idx}: {predicate_token}")

    # Iterate over tokens in the sentence
    for token_data in sentence:
        # Extract the required features:
        # 1. The predicate lemma
        # 2. Distance to the predicate
        # 3. Part of Speech (POS) as an additional feature
        word = token_data.get('features', {}).get('word', 'UNKNOWN')
        lemma = token_data.get('features', {}).get('lemma', 'UNKNOWN')
        pos_tag = token_data.get('features', {}).get('pos', 'UNKNOWN')
        dep_rel = token_data.get('features', {}).get('deprel', 'UNKNOWN')

        # Get the distance to predicate (distance is calculated based on index)
        distance = token_data.get('features', {}).get('distance_to_predicate', 'UNKNOWN')

        # Creating a dictionary with the 3 selected features
        combined_feature_dict = {
            'predicate_lemma': predicate_token['features'].get('predicate_lemma', 'UNKNOWN'),
            'distance_to_predicate': distance,
            'pos_tag': pos_tag  # You can choose another feature such as 'dep_rel' if needed
        }

        features_list.append(combined_feature_dict)  # Collect feature dicts for later fitting

    # If vectorizer is None, initialize and fit it
    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=False)
        vectorizer.fit(features_list)  # Fit the vectorizer on the features from the entire dataset

    # Apply One-Hot Encoding using DictVectorizer (this converts the dict into a one-hot encoded vector)
    for token_data, feature_dict in zip(sentence, features_list):
        one_hot_encoded_feature = vectorizer.transform([feature_dict])  # Apply transform to each token
        extracted_data.append((one_hot_encoded_feature[0].tolist(), token_data['label']))  # Convert ndarray to list

    return extracted_data, vectorizer

# Example usage: Extract features for the first sentence and predicate (just for illustration)
all_extracted_features = []
vectorizer = None  # To store the vectorizer across all sentences

# Use the function for both training and test data
for sentence in processed_data_train:  # Use the same function for training data
    features, vectorizer = extract_features(sentence, 0, vectorizer)  # Pass the entire sentence
    all_extracted_features.extend(features)

# Preview the extracted features
for feature_vector, label in all_extracted_features[:5]:  # Print first 5 examples
    print(f"Feature Vector: {feature_vector}, Label: {label}")


# Extract feature vectors and labels
X = [feature_vector for feature_vector, label in all_extracted_features]
y = [label for feature_vector, label in all_extracted_features]


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Regularization types
    'solver': ['liblinear', 'saga'],  # Solvers for optimization
    'max_iter': [100, 200, 500]  # Iterations for convergence
}

# Initialize the Logistic Regression model
model = LogisticRegression(class_weight='balanced')

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

# Fit GridSearchCV to the data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-validation Accuracy: {grid_search.best_score_}")

# Use the best model for prediction
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)


y_pred = best_model.predict(X_test)

# # Initialize and train the model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=1))


# conf_matrix = confusion_matrix(y_test, y_pred)
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.show()

# Assuming processed_data_test contains the processed test data
# Assuming each element of processed_data_test is a sentence,
# and each element of a sentence is a dictionary with 'features' and 'label' keys

flattened_data = [token for sentence in processed_data_test for token in sentence]

# Preview first token in flattened data to understand the structure
print(flattened_data[0])

# Prepare DataFrame for predictions
predictions_df = pd.DataFrame({
    'token': [token['features'].get('word', '') for token in flattened_data[:len(y_pred)]],  # Access 'word' within 'features'
    'gold_label': y_test[:len(y_pred)],
    'predicted_label': y_pred
})

# Save predictions to a TSV file
predictions_df.to_csv('model_predictions.tsv', sep='\t', index=False)

# Preview the predictions
print(predictions_df.head())

def predict_srl_labels(sentence, predicate_idx):
    # Ensure sentence format is correct (debugging)
    print(f"Sentence structure: {sentence}")

    # The original extract_features function was designed to return a tuple of
    # (feature_vector, label) for training. It needs to be modified for inference.
    # Also, it should not update the global vectorizer.

    # Update: Handle tokens with keys like 'FORM', 'LEMMA' instead of 'features'
    features, _ = extract_features(sentence, predicate_idx, vectorizer=vectorizer)  # Use existing vectorizer

    # Ensure features are extracted properly
    if len(features) == 0:
        print(f"No features extracted for sentence: {sentence}")
        return []

    # Debugging: Print features to check extraction
    print(f"Extracted features: {features}")

    # Extract feature vectors (only the features part)
    X_inference = [feature[0] for feature in features]  # No need to transform again

    # Predict labels for the current sentence
    y_inference = model.predict(X_inference)

    # Return token and predicted labels
    result = [(token['FORM'], label) for token, label in zip(sentence, y_inference)]  # Adjusted to 'FORM' instead of 'features'
    return result

# Example usage:
new_sentence = dev_parsed[0]  # Example: First sentence from dev set
predicate_index = 2  # Assume the predicate is at index 2 (adjust as necessary)

predicted_labels = predict_srl_labels(new_sentence, predicate_index)
print(predicted_labels)

