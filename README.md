# Project - Dataset for Real time pill identification with auto-labeling

This project focuses on building a comprehensive pill dataset that reflects the pills currently circulating in Vietnam. The dataset is automatically labeled and organized to support research in pill recognition and classification, especially in real-world healthcare and pharmacy scenarios.

Different deep learning models (e.g., CNN-based, Vision Transformer, lightweight architectures) are trained and evaluated on this dataset to benchmark their performance. The goal is to:
* Develop a practical pipeline for pill identification under real-world conditions.
* Support healthcare applications such as pharmacy management, counterfeit detection, and digital health assistants.
* Provide a baseline dataset and models for future research in pill recognition in Vietnam.

## Key Features
* Auto-labeling workflow for building the dataset efficiently for many different format (YOLOv8 TXT, COCO JSON, CSV).
* Dataset currently includes 22 pill with different shapes (round, oval, capsule, oblong, triangular, etc.).
* Scripts for preprocessing, augmentation, and training baseline models.
* Designed to run on real time with standard hardware while remaining extendable for large-scale research.

## Roadmap
* Expand dataset with more pill types and metadata (color, imprint, manufacturer).
* Perform EDA (Exploratory Data Analysis) to check the data integrity
* Benchmark with multiples Object Detection models (YOLO, RetinaNet, SSD, ...) models.
* Release evaluation metrics.

# Data acquisition

## Setup
We capture pill images using Iphone 11 12MP camera, the setup consist of three different background and two light source. We randomly place the pills into on the background and take 3 picture on each background with random lightning and orientation. The background is displayed below with some sample pills.

## Dataset changelog
22/09/2025 — Initialize dataset with 22 pill of different shapes (circle, oval, capsule, square, triangular, etc.). Each pill consists of 18 pictures in both front and back sides with different background, lightning and scale. 
