import pandas as pd
import numpy as np
import tensorflow as tf
import re
import pickle
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

# --- Fungsi bantu ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# --- Config ---
train_model = True  # Ubah ke True jika ingin melatih ulang
max_input_len = 20
max_response_len = 30
embedding_dim = 128
latent_dim = 256
model_path = "chatbot_ann.h5"
tokenizer_path = "tokenizer.pkl"

# --- Training Mode ---
if train_model:
    print("[INFO] Melatih model...")
    data = pd.read_csv("biaya_versi_2_raw.csv").dropna()
    data['input'] = data['input'].astype(str).apply(clean_text)
    data['response'] = data['response'].astype(str).apply(clean_text)
    data['input_with_intent'] = data['intent'] + " " + data['input']

    inputs = data['input_with_intent'].tolist()
    responses = ["<sos> " + r + " <eos>" for r in data['response'].tolist()]

    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(inputs + responses)

    input_sequences = tokenizer.texts_to_sequences(inputs)
    response_sequences = tokenizer.texts_to_sequences(responses)

    max_input_len = max(len(seq) for seq in input_sequences)
    max_response_len = max(len(seq) for seq in response_sequences)

    encoder_input = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
    decoder_input = pad_sequences(response_sequences, maxlen=max_response_len, padding='post')

    decoder_target = np.zeros_like(decoder_input)
    decoder_target[:, :-1] = decoder_input[:, 1:]

    vocab_size = len(tokenizer.word_index) + 1

    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm, state_h, state_c = LSTM(latent_dim, return_state=True)(enc_emb)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    dec_emb = Embedding(vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(dec_emb, initial_state=encoder_states)
    decoder_outputs = Dense(vocab_size, activation='softmax')(decoder_lstm)

    learning_rate = 0.001  # You can adjust this value
    optimizer = Adam(learning_rate=learning_rate)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit([encoder_input, decoder_input], np.expand_dims(decoder_target, -1),
              batch_size=64,
              epochs=400,
              validation_split=0.2)

    model.save(model_path)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"[INFO] Model dan tokenizer disimpan.")

    # --------------------------
    # 🔹 Plot Training Graphs
    # --------------------------
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.savefig("training_graph.png", dpi=300)
    plt.close()

    # --------------------------
    # 🔹 Training Summary Report
    # --------------------------
    print("\n[INFO] Training Summary Report")
    print("-" * 30)
    print(f"Total Epochs: {len(history.history['loss'])}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print("-" * 30)

else:
    print("[INFO] Memuat model dan tokenizer...")
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

# --- Intent Detection ---
intent_list = [
    'ucapan_salam_pembuka',
    'pertanyaan_seragam',
    'pertanyaan_dsp',
    'pertanyaan_pendaftaran',
    'pertanyaan_spp',
    'pertanyaan_formulir',
    'pertanyaan_sarana',
]

def detect_intent(text):
    text = clean_text(text.lower())

    # Intent ucapan salam
    if any(word in text for word in ['halo', 'assalamu', 'selamat', 'hai', 'wassalamu', 'pagi', 'siang', 'sore', 'malam', 'terima kasih']):
        return 'ucapan_salam_pembuka'
    
    # Intent pertanyaan tentang formulir
    elif any(word in text for word in ['formulir', 'harga formulir', 'formulir pendaftaran']):
        return 'pertanyaan_formulir'
    
    # Intent pertanyaan tentang SPP
    elif any(word in text for word in ['spp', 'bayar spp', 'uang bulanan', 'iuran bulanan']):
        return 'pertanyaan_spp'
    
    # Intent pertanyaan tentang biaya pendaftaran
    elif any(word in text for word in ['biaya pendaftaran', 'daftar masuk', 'uang pendaftaran', 'pendaftaran tahun ini']):
        return 'pertanyaan_pendaftaran'

    # Intent pertanyaan tentang DSP / bangunan
    elif any(word in text for word in ['dsp', 'gedung', 'bangunan', 'uang gedung', 'pembangunan']):
        return 'pertanyaan_dsp'
    
    # Intent pertanyaan tentang seragam
    elif any(word in text for word in ['seragam', 'baju sekolah', 'biaya seragam']):
        return 'pertanyaan_seragam'
    
    # Intent sarana dan prasarana
    elif any(word in text for word in ['perpustakaan', 'ruang kelas', 'kelas digital', 'ruang komputer', 'lapangan', 'uks', 'kantin', 'jumlah kelas', 'toilet', 'ac', 'mushola', 'kebersihan', 'keamanan']):
        return 'pertanyaan_sarana'
    
    # Default fallback
    else:
        return 'ucapan_salam_pembuka'


def proper_case(text):
    # Capitalize first letter of the sentence and make the rest lowercase
    # Special cases for terms like "Rp" and "SMP"
    text = text.lower()
    
    # List of abbreviations or special terms
    special_terms = ['rp', 'smp']

    words = text.split()
    for i, word in enumerate(words):
        # Capitalize the first word of the sentence
        if i == 0:
            words[i] = words[i].capitalize()
        # Keep special terms like 'Rp' and 'SMP' in uppercase
        elif word in special_terms:
            words[i] = word.upper()

    # Join back into a string
    return ' '.join(words)

def generate_response(input_text, intent):
    full_input = f"{intent} {clean_text(input_text)}"
    input_seq = tokenizer.texts_to_sequences([full_input])
    input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')

    response = ['<sos>']
    for _ in range(max_response_len):
        token_seq = tokenizer.texts_to_sequences([response])[0]
        token_seq = pad_sequences([token_seq], maxlen=max_response_len, padding='post')

        predictions = model.predict([input_seq, token_seq], verbose=0)
        next_token_id = np.argmax(predictions[0, len(response)-1])
        next_word = tokenizer.index_word.get(next_token_id, '')
        if next_word == '<eos>' or not next_word:
            break
        response.append(next_word)

    # Join the response and remove <sos> and <eos>
    final_response = ' '.join(response[1:])
    final_response = final_response.replace('<eos>', '').strip()  # Remove <eos> token
    return proper_case(final_response)