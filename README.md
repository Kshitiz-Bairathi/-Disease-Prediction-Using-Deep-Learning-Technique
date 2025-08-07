# Disease-Prediction-Using-Deep-Learning-Technique.


First of all ,I would like to express my sincere gratitude to my project partner Kshitiz Bairathi for his invaluable collaboration and dedication throughout this project. Special thanks to our mentor Dr. Manoj Kumar for his expert guidance, continuous support, and insightful feedback that made this work possible.


 # Overview
This project implements automated dementia detection using Convolutional Neural Networks (CNNs) to analyze MRI brain scans. The system can classify images into four distinct stages of Alzheimer's disease progression, providing a valuable tool for early diagnosis and monitoring.

 # Dataset

Size: 44,000 MRI brain scan images
Classes: 4 categories

1.NonDemented (12,800 images)
2.VeryMildDemented (11,200 images)
3.MildDemented (10,000 images)
4.ModerateDemented (10,000 images)


Format: JPEG, 180x180 pixels, RGB.
Preprocessing: Skull-stripped, centered brain structure, clean black background.

 # Key Features

1.Multiple Architecture Support: Custom CNNs and pre-trained models
2.Transfer Learning: Fine-tuned pre-trained networks for medical imaging
3.Comprehensive Evaluation: Multiple performance metrics and visualization
4.Scalable Design: Optimized for deployment in healthcare settings
5.Cost-Performance Analysis: Models optimized for different resource constraints

 # Model Architectures

# Custom CNN Models (Built from Scratch)

1.7-13 Convolutional Layers: Various depths tested for optimal performance
2.Range: 7-layer (93.30%) to 13-layer (96.50%) CNN models
3.Best Custom Model: 10-layer CNN with 96.63% accuracy
4.Optimal Performance: 9-layer CNN achieved 96.50% accuracy with efficient architecture

# Pre-trained Models

1.VGG16: 95.16% accuracy
2.ResNet50: 92.80% accuracy
3.MobileNetV2: 96.34% accuracy
4.DenseNet-121: 98.01% accuracy (Best Performance)

 # Technologies Used

1.Deep Learning Framework: PyTorch
2.Programming Language: Python
3.Computer Vision: OpenCV, PIL
4.Data Analysis: NumPy, Pandas
5.Visualization: Matplotlib, Seaborn
6.Metrics: Scikit-learn
