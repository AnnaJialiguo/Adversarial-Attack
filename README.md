# Jailbreaking Deep Models 🔓🧠

This repository contains my implementation and analysis of adversarial attacks on pretrained deep image classifiers, as part of my Deep Learning coursework (Spring 2025).

## 📌 Overview

We study the robustness of ResNet-34 under different $L_\infty$-bounded adversarial attacks, including:

- **FGSM** (Fast Gradient Sign Method)
- **PGD** (Projected Gradient Descent)
- **Patch Attack** (localized region only)

We also evaluate **transferability** of these attacks to a separate model, **DenseNet-121**.

## 📁 Project Structure

- `project3.ipynb`: Full implementation, experiment results, and figures.
- `src/`: Modular attack code and utilities.
- `results/`: Accuracy logs and adversarial images.
- `TestDataSet/`: Folder structure sample (you must download data externally).

## 📊 Key Results

| Model          | Dataset        | Top-1 Acc | Top-5 Acc |
|----------------|----------------|-----------|-----------|
| ResNet-34      | Original       | 76.0%     | 94.2%     |
| ResNet-34      | FGSM           | 26.6%     | 50.4%     |
| ResNet-34      | PGD            | 0.4%      | 6.8%      |
| ResNet-34      | Patch          | 67.4%     | 90.6%     |
| DenseNet-121   | FGSM           | 42.6%     | 66.2%     |
| DenseNet-121   | PGD            | 39.0%     | 63.6%     |
| DenseNet-121   | Patch          | 72.6%     | 92.4%     |

## 📚 Acknowledgements

- Based on NYU Deep Learning (Spring 2025) Project 3.
- Thanks to Prof. Chinmay Hegde for the course materials.

## 📝 License

This project is for academic demonstration purposes only.
