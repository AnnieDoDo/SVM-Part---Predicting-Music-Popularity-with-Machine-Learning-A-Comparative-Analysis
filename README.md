## 🎵 Predicting Music Popularity with SVM 
### Python, IEEE Big Data Workshop 2023 – accepted
Dataset: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data)

- Built a custom Support Vector Machine from scratch to classify music tracks based on genre, tempo, key, and other features.
- Avoided high-level ML libraries to focus on core algorithm implementation and hyperparameter tuning (C, kernel functions).
- Compared hand-crafted SVM to scikit-learn and visualized decision boundaries.
- Paper accepted to IEEE Big Data 2023 Workshop (not presented).

### 🔧 with Extented with GPU-Accelerated Linear SVM
To demonstrate the impact of low-level optimization, we extended the linear SVM by implementing custom CUDA kernels using Numba. This allowed us to parallelize the training loop — especially the hinge loss gradient computation — across thousands of GPU threads.

**Highlights:**
- CUDA-based gradient computation via `@cuda.jit`
- Parallel execution using 2D grid strategy
- Achieved 2–4x speedup on training steps for medium datasets

> This project was originally implemented from scratch to gain deeper insight into the optimization process behind SVM. <br>
> GPU acceleration was later added to simulate the real-world demand for faster ML training in low-latency applications.
