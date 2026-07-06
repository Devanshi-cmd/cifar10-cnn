# CIFAR-10 Image Classifier — Custom CNN

A convolutional neural network built **from scratch in PyTorch** (no pretrained weights) to classify images into 10 categories from the CIFAR-10 dataset. Includes an interactive Streamlit app for real-time inference on any uploaded image.

**Live demo:** https://cifar10-cnn-jtqevsx42ezhv8uqgw3jxz.streamlit.app

## Overview

- 3-block CNN architecture designed and trained from scratch
- ~85% test accuracy on CIFAR-10
- BatchNorm, Dropout, and data augmentation for regularization
- Cosine Annealing learning rate scheduling
- Deployed as a live web app via Streamlit Community Cloud

## Classes

`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

## Project Structure

```
cifar10-cnn/
├── app.py                # Streamlit web app for inference
├── model.py               # CNN architecture definition
├── train.py                # Training script
├── cnn_cifar10.pth        # Trained model weights
├── requirements.txt        # Python dependencies
└── README.md
```

## Running Locally

```bash
git clone https://github.com/Devanshi-cmd/cifar10-cnn.git
cd cifar10-cnn
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal, upload an image, and view the predicted class with confidence scores and a probability breakdown across all 10 classes.

## Training

If you want to retrain the model from scratch:

```bash
python train.py
```

This will train the CNN on the CIFAR-10 dataset and save weights to `cnn_cifar10.pth`.

## Tech Stack

- **PyTorch** — model definition and training
- **Streamlit** — interactive web interface
- **Matplotlib** — probability visualization
- **PIL / torchvision** — image preprocessing

## Author

**Devanshi Tiwari**
[GitHub](https://github.com/Devanshi-cmd) · [LinkedIn](https://linkedin.com/in/devanshi-tiwari-8ab247320)
