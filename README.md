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

🛠️ System Architecture
Data Perception: Background standardization and noise suppression for microscopic imagery.

Core Inference: Improved YOLOv11 weights optimized for thin filament-like structures.

Quantification Logic: Mapping 2D pixel masks to equivalent biomass using the UCAB (Unit Cell Area Benchmark) model.

Result Visualization: Real-time rendering of detection boxes and population structure pie charts.

📂 Repository Structure
├── models/             # Pre-trained ONNX weights (DVB + BiFPN)
├── src/
│   ├── ui/             # PyQt5 interface designs
│   ├── engine/         # ONNX inference and post-processing
│   └── utils/          # Density calculation and XML export logic
├── data/               # Sample microscopic images (Dianchi Huiwan)
├── gui_main.py         # Main entry point for the software
└── requirements.txt    # Dependency list

<img width="3062" height="1245" alt="P-R对比" src="https://github.com/user-attachments/assets/9a1a72ae-d901-48e7-a0a2-d267d0ff6ec7" />

🚀 Quick Start
Prerequisites
Python 3.9+

CUDA 11.x (Optional, for GPU acceleration)

📈 Real-world Application
The system was validated using field samples from Dianchi Lake (Huiwan Site) spanning from Nov 2024 to Jul 2025. It successfully quantified species such as Microcystis and Ankistrodesmus even under bloom conditions with densities reaching 3.85e8 Cells/L.
<img width="2385" height="958" alt="algae_comparison_chart_v2" src="https://github.com/user-attachments/assets/4346489c-3ad9-4f70-9d75-b59f95547e81" />
