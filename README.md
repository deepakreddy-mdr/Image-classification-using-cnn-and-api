Project Title

Image Classification Using Convolutional Neural Networks with AI Comparison

1. Project Overview

This project is designed to automatically classify images using a Convolutional Neural Network (CNN) and compare the prediction with a cloud-based AI model (Google Gemini).

A web application was developed using Flask, where users can upload an image and view predictions from both models.

The system performs:

Image upload

Preprocessing

CNN prediction

Gemini AI prediction

Display results with confidence and comparison

2. Objective of the Project

The main objectives are:

Build a CNN model for image classification

Predict objects in unseen images

Compare results with a pretrained AI model

Provide an interactive web interface

3. Technologies Used
Component	Technology
Programming Language	Python
Deep Learning	TensorFlow / Keras
Image Processing	OpenCV
Web Framework	Flask
Cloud AI	Google Gemini API
Frontend	HTML, Tailwind CSS
Dataset	CIFAR-10
Numerical Computing	NumPy
4. System Architecture

The workflow of the system:

User Uploads Image
        ↓
Image Preprocessing
        ↓
CNN Model Prediction
        ↓
Gemini AI Prediction
        ↓
Display Results in Web Interface

5. How the System Works
Step 1: Image Upload

The user uploads an image using a web interface built with Flask.

Step 2: Preprocessing

The image is:

Read using OpenCV

Resized to required dimensions

Normalized for CNN input

Step 3: CNN Prediction

The trained CNN model:

Extracts features

Classifies the image

Displays top predictions with confidence

Step 4: Gemini AI Prediction

The same image is sent to Gemini API, which:

Analyzes the image using a pretrained model

Returns an object description

Step 5: Display Results

The web application shows:

Uploaded image

CNN prediction

Top 3 predictions with confidence

Gemini prediction

Comparison table

6. CNN Working Principle

CNN works in multiple stages:

Convolution Layer

Extracts features like edges and shapes

Pooling Layer

Reduces image size

Keeps important information

Flatten Layer

Converts features into a vector

Dense Layer

Performs classification

7. Dataset Used

The CNN model was trained using:

CIFAR-10 dataset

10 object classes

Small colored images

The dataset helps the model learn patterns of objects.

8. Results

The system successfully:

Classifies images locally using CNN

Uses Gemini AI for additional validation

Displays predictions in real time

The comparison helps evaluate model performance.

9. Features of the Project

This project includes:

Image upload interface

CNN prediction

Top 3 predictions

Confidence visualization

Gemini AI integration

Comparison table

Modern UI

These features make the project more advanced than a basic CNN implementation.

10. Applications

This system can be used in:

Medical image analysis

Security systems

Autonomous vehicles

Traffic sign recognition

Quality inspection

11. Limitations

Some limitations include:

CNN accuracy depends on training data

Internet is required for Gemini API

Performance may vary with low-quality images

12. Future Enhancements

Possible improvements:

Real-time webcam detection

Larger datasets for better accuracy

Mobile application

Deployment on cloud

Object detection instead of classification

13. Conclusion

This project demonstrates an end-to-end image classification system that combines:

Deep learning

Cloud AI

Web development

The system provides accurate predictions and an interactive interface, showing how modern AI applications are built.
