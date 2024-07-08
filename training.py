import random
import json
import pickle
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize patterns
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize training data
training = []
outputEmpty = [0] * len(classes)

# Create bag of words for each document
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
    # One-hot encode the output
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffle and convert training to numpy array
random.shuffle(training)
training = np.array(training)

# Split into input and output
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])

# Compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')
print('Done')
