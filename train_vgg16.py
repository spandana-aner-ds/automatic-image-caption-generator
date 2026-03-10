from models.vgg16_lstm_model import build_vgg16_lstm
from utils.preprocess import *
from utils.dataset_loader import *
import numpy as np

captions = load_captions("data/Flickr8k_text/Flickr8k.token.txt")
captions = clean_captions(captions)

tokenizer = build_tokenizer(captions)
save_tokenizer(tokenizer, "tokenizer.pkl")

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for caps in captions.values() for c in caps)

features = load_features("features/vgg16_features.pkl")

model = build_vgg16_lstm(vocab_size, max_length)
model.compile(loss="categorical_crossentropy", optimizer="adam")

X1, X2, y = create_sequences(tokenizer, captions, features, max_length, vocab_size)

model.fit([X1, X2], y, epochs=20, batch_size=32)
model.save("saved_models/vgg16_model.h5")
