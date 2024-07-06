# import libraries
import tkinter as tk
from tkinter import ttk
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import re
import tensorflow.strings as tf_strings
import json
import string
from keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import numpy as np
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import playsound
from datetime import datetime
import pytz

# Initialize recognizer class (for recognizing the speech)
recognizer = sr.Recognizer()
translator = Translator()

# English to Spanish translation
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf_strings.lower(input_string)
    return tf_strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")

# Load the English vectorization layer configuration
with open('eng_vectorization_config.json') as json_file:
    eng_vectorization_config = json.load(json_file)

# Recreate the English vectorization layer with basic configuration
eng_vectorization = TextVectorization(
    max_tokens=eng_vectorization_config['max_tokens'],
    output_mode=eng_vectorization_config['output_mode'],
    output_sequence_length=eng_vectorization_config['output_sequence_length']
)

# Apply the custom standardization function
eng_vectorization.standardize = custom_standardization

# Load the Spanish vectorization layer configuration
with open('spa_vectorization_config.json') as json_file:
    spa_vectorization_config = json.load(json_file)

# Recreate the Spanish vectorization layer with basic configuration
spa_vectorization = TextVectorization(
    max_tokens=spa_vectorization_config['max_tokens'],
    output_mode=spa_vectorization_config['output_mode'],
    output_sequence_length=spa_vectorization_config['output_sequence_length'],
    standardize=custom_standardization
)

# Load and set the English vocabulary
with open('eng_vocab.json') as json_file:
    eng_vocab = json.load(json_file)
    eng_vectorization.set_vocabulary(eng_vocab)

# Load and set the Spanish vocabulary
with open('spa_vocab.json') as json_file:
    spa_vocab = json.load(json_file)
    spa_vectorization.set_vocabulary(spa_vocab)

# Load the Spanish model
transformer = load_model('transformer_model')

spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sentence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = tf.argmax(predictions[0, i, :]).numpy().item(0)
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence.strip().replace("[start]", "").replace("[end]", "")

# English to French translation

# Load the French model
model = load_model('english_to_french_model')

# Load Tokenizer
with open('english_tokenizer.json') as f:
    data = json.load(f)
    english_tokenizer = tokenizer_from_json(data)

with open('french_tokenizer.json') as f:
    data = json.load(f)
    french_tokenizer = tokenizer_from_json(data)

# Load max length
with open('sequence_length.json') as f:
    max_length = json.load(f)

def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def translate_to_french(english_sentence):
    english_sentence = english_sentence.lower()
    english_sentence = english_sentence.replace(".", '')
    english_sentence = english_sentence.replace("?", '')
    english_sentence = english_sentence.replace("!", '')
    english_sentence = english_sentence.replace(",", '')
    
    english_sentence = english_tokenizer.texts_to_sequences([english_sentence])
    english_sentence = pad(english_sentence, max_length)
    
    english_sentence = english_sentence.reshape((-1, max_length))
    
    french_sentence = model.predict(english_sentence)[0]
    
    french_sentence = [np.argmax(word) for word in french_sentence]
    french_sentence = french_tokenizer.sequences_to_texts([french_sentence])[0]
    
    return french_sentence.replace("[start]", "").replace("[end]", "").strip()

def translate_to_spanish(english_sentence):
    spanish_sentence = decode_sentence(english_sentence)
    return spanish_sentence.strip()

# Function to handle translation request based on selected language
def handle_translate():
    selected_language = language_var.get()
    english_sentence = text_input.get("1.0", "end-1c")
    
    if selected_language == "French":
        translation = translate_to_french(english_sentence)
    elif selected_language == "Spanish":
        translation = translate_to_spanish(english_sentence)
        
    translation_output.delete("1.0", "end")
    translation_output.insert("end", f"{selected_language} translation: {translation}")

def is_valid_time():
    # Get the current time in IST
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    # Check if current time is after 6 PM
    if current_time.hour >= 18:
        return True
    return False

def translate_audio_to_hindi(audio_text):
    words = audio_text.split()
    translated_words = []
    for word in words:
        if word.lower().startswith(('m', 'o')):
            continue
        translated_word = translator.translate(word, src='en', dest='hi').text
        translated_words.append(translated_word)
    return ' '.join(translated_words)

def handle_audio_translate():
    if not is_valid_time():
        translation_output.delete("1.0", "end")
        translation_output.insert("end", "Please try after 6 PM IST")
        return

    with sr.Microphone() as source:
        translation_output.delete("1.0", "end")
        translation_output.insert("end", "Speak now...")
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            translation_output.delete("1.0", "end")
            translation_output.insert("end", f"You said: {text}")
            translated_text = translate_audio_to_hindi(text)
            translation_output.insert("end", f"\nHindi translation: {translated_text}")
            tts = gTTS(translated_text, lang='hi')
            tts.save("translated_audio.mp3")
            playsound.playsound("translated_audio.mp3")
        except sr.UnknownValueError:
            translation_output.delete("1.0", "end")
            translation_output.insert("end", "Sorry, could not understand the audio. Please repeat.")
        except sr.RequestError as e:
            translation_output.delete("1.0", "end")
            translation_output.insert("end", f"Could not request results; {e}")

# Setting up the main window
root = tk.Tk()
root.title("Language Translator")
root.geometry("550x600")

# Font configuration
font_style = "Times New Roman"
font_size = 14

# Frame for input
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

# Heading for input
input_heading = tk.Label(input_frame, text="Enter the text to be translated", font=(font_style, font_size, 'bold'))
input_heading.pack()

# Text input for English sentence
text_input = tk.Text(input_frame, height=5, width=50, font=(font_style, font_size))
text_input.pack()

# Language selection
language_var = tk.StringVar()
language_label = tk.Label(root, text="Select the language to translate to", font=(font_style, font_size, 'bold'))
language_label.pack()
language_select = ttk.Combobox(root, textvariable=language_var, values=["French", "Spanish"], font=(font_style, font_size), state="readonly")
language_select.pack()

# Submit button
submit_button = ttk.Button(root, text="Translate Text", command=handle_translate)
submit_button.pack(pady=10)

# Button for audio translation
audio_button = ttk.Button(root, text="Translate Audio to Hindi", command=handle_audio_translate)
audio_button.pack(pady=10)

# Frame for output
output_frame = tk.Frame(root)
output_frame.pack(pady=10)

# Heading for output
output_heading = tk.Label(output_frame, text="Translation: ", font=(font_style, font_size, 'bold'))
output_heading.pack()

# Text output for translations
translation_output = tk.Text(output_frame, height=10, width=50, font=(font_style, font_size))
translation_output.pack()

# Running the application
root.mainloop()
