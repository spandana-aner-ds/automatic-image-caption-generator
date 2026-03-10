# Automatic Image Caption Generator using Deep Learning

## 📌 Project Overview

This project builds a deep learning model that automatically generates captions for images. It combines **CNN (VGG16)** for extracting image features and **LSTM** for generating natural language captions.

## 🚀 Features

* Image feature extraction using VGG16
* Caption generation using LSTM
* Deep Learning based image understanding
* Automatic caption prediction for new images

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* Deep Learning
* CNN (VGG16)
* LSTM
* Natural Language Processing

## 📂 Project Files

* `train_vgg16.py` – Extract image features
* `train_caption_model.py` – Train caption generation model
* `vgg16_lstm_model.py` – Model architecture
* `app.py` – Generate captions for images

## 📂 Dataset

The dataset used for this project is **Flickr8k Dataset**, which contains 8000 images with captions.

Due to its large size, the dataset is not uploaded to this repository.

You can download the dataset from the link below:

Dataset Link: https://www.kaggle.com/datasets/adityajn105/flickr8k

After downloading, place the dataset folder inside the project directory.

## ▶️ How to Run

Install dependencies:
pip install -r requirements.txt

Train the model:
python train_caption_model.py

Run the caption generator:
python app.py

## Output
Input Image: dog.jpg
Generated Caption: "A dog running in the grass."

## 👩‍💻 Author

Spandana | Data Science Enthusiast
