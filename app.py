import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

total_words = len(tokenizer.word_index) + 1
max_sequence_len = 14  # matches training

# Rebuild the exact same architecture used in training
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words, activation="softmax"))

# Load only the trained weights
model.load_weights("next_word_lstm.weights.h5")

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word Prediction With LSTM")
input_text = st.text_input("Enter the sequence of Words", "To be or not to")
if st.button("Predict Next Word"):
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')