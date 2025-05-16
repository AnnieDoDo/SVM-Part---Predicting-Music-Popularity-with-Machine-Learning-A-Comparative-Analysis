# 🎵 Predicting Music Popularity with Machine Learning

This project aims to predict the popularity or genre of music tracks using Support Vector Machines (SVM), implemented from scratch with additional GPU acceleration using CUDA. The work explores the full modeling pipeline — from data processing to training and evaluation — and was accepted for presentation at the IEEE Big Data 2023 Workshop.

---

## 📊 Dataset

The dataset used is the [Spotify Tracks Dataset (1921–2020)](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-160k-tracks), containing over 160,000 tracks and metadata such as:

- Danceability
- Energy
- Key
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Genre
- Popularity Score

After cleaning, we selected 10 core features and converted categorical labels into binary targets for classification.

---

## 🧠 Methodology

We implemented three different models to understand performance trade-offs:

### 1. **Custom Linear SVM**
- Implemented from scratch using NumPy.
- Supports gradient descent with hinge loss and L2 regularization.
- Offers clear visibility into model mechanics and optimization.

### 2. **scikit-learn SVM**
- Served as a baseline using `sklearn.svm.SVC`.
- Utilized cross-validation for model robustness assessment.

### 3. **GPU-Accelerated SVM**
- Rewrote the weight and bias update functions using `Numba` and `CUDA`.
- Leveraged parallel execution to accelerate gradient computation.

---

## 🚀 GPU Acceleration with CUDA

To reduce training time on larger datasets, we extended our custom SVM with GPU-based gradient updates:

- Used `@cuda.jit` to define parallel kernels for weight and bias gradients.
- Used `cuda.atomic.add` to avoid race conditions in parallel reduction.
- Achieved **2–4x speedup** on 10,000+ row datasets versus CPU.

> This section bridges ML implementation with systems-level optimization, simulating a real-world need for scalable, high-throughput training pipelines in quantitative analysis or streaming data settings.

---

## 📈 Evaluation & Results
We set Linear Regression as benchmark.

| Model         | Accuracy | MSE     | MAE     | MAPE    |
|---------------|----------|---------|---------|---------|
| Linear Regression | - | 484.55 | 18.36 | 2.09*10^16%  |
| scratch SVM   | 0.9912280701754386    | 0.03737  | 0.01868 | 1.86842%  |


Additional metrics were computed using cross-validation and test sets. We also measured wall time for training using the GPU vs CPU to highlight performance gain.

---

