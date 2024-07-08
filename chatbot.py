import random
import json
import pickle
import numpy as np
import nltk
import streamlit as st
import tensorflow as tf
import time
import os

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Define the path to the intents.json file
file_path = os.path.join(os.path.dirname(__file__), 'intents.json')

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file '{file_path}' does not exist.")

# Load intents from JSON file
with open(file_path, encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')

# Initialize session state
if "greeting_displayed" not in st.session_state:
    st.session_state.greeting_displayed = False

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list




def get_response(intents_list, intents_json):
    if intents_list:  # Check if intents_list is not empty
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "Sorry, I don't understand that."


def main():
    st.set_page_config(
        page_title="Factogram.ai",
        page_icon="👨🏼‍🔬",
        layout="wide",
    )

    st.title("👨🏼‍🔬 Factogram.ai")

    # Initialize session state
    if "greeting_displayed" not in st.session_state:
        st.session_state.greeting_displayed = False

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display greeting message from HistBot
    if not st.session_state.greeting_displayed:
        histbot_greeting = "Hello! I'm Factogram.ai How can I assist you today?"
        st.session_state.messages.append(
            {"role": "assistant", "content": histbot_greeting})
        st.session_state.greeting_displayed = True

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.text_input(
                "You:", value=message["content"], key=message["content"])
        elif message["role"] == "assistant":
            st.text("SBot: " + message["content"])  # Concatenate strings here

    # Accept user input
    prompt = st.text_input("Message Facto Bot")

    # Add user message to chat history
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        st.text_input("You:", value=prompt, key=prompt)

        # Display assistant response in chat message container
        message_placeholder = st.empty()
        full_response = ""
        # Predicting intent and getting a response
        ints = predict_class(prompt)
        assistant_response = get_response(ints, intents)
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.text("SciBot:\n" + full_response + "▌")
        message_placeholder.text("SciBot:\n" + full_response)
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})


# Run the Streamlit app
if __name__ == "__main__":
    main()
