# Phytoplankton-Detection-System
基于深度学习的浮游植物智能检测系统开发及其应用

AlgaeSmartDet: Intelligent Phytoplankton Detection and Quantification System
This repository contains the official implementation of an intelligent monitoring system for phytoplankton, featuring a deep-learning-based detection engine and a PyQt5-driven graphical user interface. The system is designed to handle complex microscopic water samples with high abundance and overlapping cells.

🌟 Key Features
Advanced Detection Engine: Powered by an improved YOLOv11 architecture with DVB (Dynamic Vector Boundary) and BiFPN (Bidirectional Feature Pyramid Network).

High Precision: Achieves 97.3% mAP@50 on laboratory datasets and maintains robust performance in wild habitats like Dianchi Lake.

Dual Mode Operation: Supports both single-image interactive analysis and high-throughput batch monitoring.

Biomass Quantification: Includes a pixel-level equivalent counting model to correct estimation errors caused by cell overlapping.

Seamless Deployment: Models are exported via ONNX for high-speed inference on standard hardware (up to 239.7 FPS).
