# Facial-Emotion-Recognition-Using-Deep-Learning

### Project Summary
This project, titled "DeepFER" (Deep Facial Emotion Recognition), is a deep learning-based system designed to recognize human emotions from facial expressions in real-time. It leverages the power of Convolutional Neural Networks (CNNs) and Transfer Learning to classify facial images into one of seven fundamental emotion categories: Angry, Sad, Happy, Fear, Neutral, Disgust, and Surprise.

The system is built with robustness in mind, utilizing a diverse dataset of both posed and spontaneous expressions. To enhance model generalization and accuracy, various data augmentation techniques such as rotation, flipping, and zooming are applied during the data preprocessing phase.

The ultimate goal of DeepFER is to enable practical applications in a wide range of fields, including mental health monitoring, customer service feedback analysis, and advanced human-computer interaction, where the ability to accurately recognize emotions can significantly improve user experience and system responsiveness.

### Features
- Real-time Recognition: Uses a live webcam feed to perform real-time emotion classification.

- Multiple Models: Includes several pre-trained models, such as custom CNNs and a transfer-learned EfficientNet model, allowing for flexible use and experimentation.

- Streamlit Web Interface: A user-friendly web application built with Streamlit provides an easy-to-use interface for demonstration and testing.

- Comprehensive File Structure: The repository is well-organized with dedicated folders for original and preprocessed images, making it easy to understand the project's data flow.

- Jupyter Notebooks: An accompanying Jupyter notebook (model_testing.ipynb) details the complete model training, evaluation, and fine-tuning process.

### Technology Stack
- Deep Learning: TensorFlow, Keras

- Frontend: Streamlit

- Image Processing: OpenCV

- Language: Python

### File Structure
Facial-Emotion-Recognition-Using-Deep-Learning/
├── original_images/                                                # Raw images for model training
├── preprocessed_images/                                            # Processed images after data augmentation
├── app.py                                                          # Main Streamlit application script
├── best_emotion_model.keras                                        # The final, best-performing model
├── cnn_custom_first_model.keras                                    # Example of a custom CNN model
├── final_transfer_learning_emotion_recognition_model.keras
├── Transfer_learning_efficientnet_model.keras                      # Model trained with transfer learning
├── model_testing.ipynb                                             # Jupyter Notebook for model development
├── requirements.txt                                                # List of all Python dependencies
├── README.md                                                       # This file
└── DeepFER: Facial Emotion Recognition Using Deep Learning.ipynb   # Main jupterfile for experimentation and model building

### Contribution
Individual Contributor: Mritunjay Mishra
