# Ai_Image_Distinguish_Model

Ai_Image_Distinguish_Model is an AI-based image detection model designed to distinguish between real images and AI-generated or deepfake images. The model is trained using a Swin Transformer architecture, which is powerful for understanding image patterns, textures, and visual details.

This model is part of **CyberShield AI**, a Final Year Project focused on protecting users from digital threats using artificial intelligence.

## Project Overview

The main purpose of this model is to detect whether an image is real or AI-generated. It analyzes visual patterns, fake textures, unnatural details, and image artifacts that are commonly found in AI-generated images.

Instead of using traditional image classification methods, this project uses a **Swin Transformer model** because it provides better image understanding through attention-based learning.

## Key Features

- Detects real and AI-generated images
- Uses Swin Transformer for image classification
- Identifies fake textures and unnatural visual patterns
- Supports deepfake and AI-image detection
- Trained on real and fake image datasets
- Can be integrated with backend APIs or mobile applications
- Useful for cybersecurity and digital content verification

## Model Training

The model was trained using real and AI-generated image datasets. The training process included:

1. Collecting real and fake image data
2. Cleaning and organizing the dataset
3. Splitting data into training and validation sets
4. Applying image preprocessing such as resizing and normalization
5. Loading a Swin Transformer model
6. Fine-tuning the model for binary classification
7. Training the model to classify images as real or fake
8. Evaluating performance using accuracy and validation results
9. Saving the trained model for future prediction and API integration

## How It Works

The model takes an image as input and predicts whether the image is real or AI-generated.

Example input:

```text
Uploaded image
