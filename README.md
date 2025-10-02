# Project - Dataset for Real time pill identification with auto-labeling

This project focuses on building a comprehensive pill dataset that reflects the pills currently circulating in Vietnam. The dataset is automatically labeled and organized to support research in pill recognition and classification, especially in real-world healthcare and pharmacy scenarios.

Different deep learning models (e.g., CNN-based, Vision Transformer, lightweight architectures) are trained and evaluated on this dataset to benchmark their performance. The goal is to:
* Develop a practical pipeline for pill identification under real-world conditions.
* Support healthcare applications such as pharmacy management, counterfeit detection, and digital health assistants.
* Provide a baseline dataset and models for future research in pill recognition in Vietnam.

## Key Features
* Auto-labeling workflow for building the dataset efficiently for many different format (YOLOv8 TXT, COCO JSON, CSV).
* Dataset currently includes 378 pill with different shapes (round, oval, capsule, oblong, triangular, etc.).
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

## Dataset access
This dataset can be accessed through Kaggle with following link: https://www.kaggle.com/datasets/cistily/vn-pills-original

## Dataset changelog
01/10/2025 — Added 121 pills with 2,006 images.
➡️ Current total: 378 pills, 6,632 images.

29/09/2025 — Added 36 pills with 648 images.
➡️ Current total: 257 pills, 4,626 images.

27/09/2025 — Added 55 pills with 990 images.
➡️ Current total: 221 pills, 3,978 images.

25/09/2025 — Added 68 pills with 1,224 images.
➡️ Current total: 165 pills, 2,988 images.

24/09/2025 — Added 48 pills with 864 images.
➡️ Current total: 98 pills, 1,764 images.

23/09/2025 — Added 28 pills with 504 images.
➡️ Current total: 50 pills, 900 images.

22/09/2025 — Dataset initialized with 22 pills of various shapes (circle, oval, capsule, square, triangular, etc.). ➡️ Current total: 22 pills, 396 images.
* Each pill includes 18 images (front & back) under different backgrounds, lighting, and scales.
