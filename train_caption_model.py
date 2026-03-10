# COMMON TRAINING SCRIPT (MEMORY-SAFE) FOR ALL CNN+LSTM MODELS
# Uses tf.data.Dataset to avoid OOM and generator errors

import os
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FEATURE_DIR = os.path.join(BASE_DIR, 'features')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

MODEL_NAME = 'inceptionv3'   # change later for other models
# vgg16 | resnet50 | inceptionv3 | efficientnetb0 | mobilenetv2

EPOCHS = 5
BATCH_SIZE = 32
MAX_LENGTH = 37

# ================= LOAD TOKENIZER =================
with open(os.path.join(DATA_DIR, 'preprocessed', 'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1

# ================= LOAD FEATURES =================
with open(os.path.join(FEATURE_DIR, f'{MODEL_NAME}_features.pkl'), 'rb') as f:
    features = pickle.load(f)

# ================= LOAD CAPTIONS =================
captions = {}
with open(os.path.join(DATA_DIR, 'Flickr8k.token'), 'r') as f:
    for line in f:
        image_id, caption = line.strip().split('\t')
        image_id = image_id.split('.')[0]
        caption = 'startseq ' + caption + ' endseq'
        captions.setdefault(image_id, []).append(caption)

# ================= DATA GENERATOR =================
def caption_generator():
    while True:
        for image_id, caps in captions.items():
            if image_id not in features:
                continue
            feature = features[image_id][0].astype(np.float32)

            for cap in caps:
                seq = tokenizer.texts_to_sequences([cap])[0]

                for i in range(1, len(seq)):
                    in_seq = pad_sequences([seq[:i]], maxlen=MAX_LENGTH)[0]
                    out_seq = to_categorical(seq[i], num_classes=vocab_size)

                    yield (
                        (feature, in_seq.astype(np.int32)),
                        out_seq.astype(np.float32)
                    )

# ================= TF DATASET =================
dataset = tf.data.Dataset.from_generator(
    caption_generator,
    output_signature=(
        (
            tf.TensorSpec(shape=(features[next(iter(features))][0].shape[0],), dtype=tf.float32),
            tf.TensorSpec(shape=(MAX_LENGTH,), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32),
    ),
)

dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ================= LOAD MODEL =================
if MODEL_NAME == 'vgg16':
    from models.vgg16_lstm_model import build_vgg16_lstm as build_model
elif MODEL_NAME == 'resnet50':
    from models.resnet50_lstm_model import build_resnet50_lstm as build_model
elif MODEL_NAME == 'inceptionv3':
    from models.inceptionv3_lstm_model import build_inceptionv3_lstm as build_model
elif MODEL_NAME == 'efficientnetb0':
    from models.efficientnetb0_lstm_model import build_efficientnetb0_lstm as build_model
elif MODEL_NAME == 'mobilenetv2':
    from models.mobilenetv2_lstm_model import build_mobilenetv2_lstm as build_model
else:
    raise ValueError("Invalid MODEL_NAME")

model = build_model(vocab_size, MAX_LENGTH)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# ================= TRAIN =================
steps_per_epoch = 6000  # safe fixed value for CPU

checkpoint_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}_lstm.keras')
checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', save_best_only=True)

model.fit(
    dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

print("✅ Training completed. Model saved at:", checkpoint_path)
