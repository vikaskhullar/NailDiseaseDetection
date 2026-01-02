Title
Low Resources Devices Based Federated Learning for Nail Disease Detection

Abstract

Background: Nail diseases, encompassing prevalent ailments such as fungal infections and more severe problems like melanoma, can serve as significant indications of general health and necessitate precise diagnosis for appropriate treatment.

Objective: The paper focuses on the development of a nail disease detection system using advanced machine learning techniques, combining transfer learning and federated learning frameworks. The study aims to demonstrate the potential of integrating machine learning and federated learning for nail disease detection, achieving high performance without data sharing.

Methods: The dataset comprises images of various nail conditions such as Acral Lentiginous Melanoma, Onychogryphosis, and Pitting, among others, which are processed to ensure consistent quality for robust model training. Key models used for feature extraction include ResNet152V2, DenseNet201, MobileNetV2, and InceptionResNetV2, which generate feature sets ranging from 1,280 to 2,048 features per image. These features are then combined to form a unified feature space of 6,784 dimensions, further reduced to five key features using Linear Discriminant Analysis (LDA) for efficient classification.

Results: Various classification models, such as Deep Neural Networks (DNN), Long Short-Term Memory (LSTM), and Bidirectional LSTM (Bi-LSTM), are evaluated, with the Bi-LSTM achieving the highest classification accuracy of 91.8%. The federated learning approach facilitates collaborative model training across multiple clients while preserving data privacy, achieving validation accuracy rates of over 99% in both independent and identically distributed (IID) and non-IID data environments.

Conclusions: The proposed federated learning based models resulted high in both IID and non-IID data distributions.

Data Avaialble at: https://www.kaggle.com/datasets/nikhilgurav21/nail-disease-detection-dataset
