# Handwritten Digit Recognition using Support Vector Machine (SVM)

---

## Abstract

Have you ever wondered how a computer figures out what number you just scribbled? That's exactly the question this project tries to answer — and more importantly, actually solve in a way you can interact with in real time.

This project is a desktop application that lets you draw a digit (or two) on a canvas with your mouse, hit a button, and watch the system tell you what it thinks you wrote — along with how confident it is. Under the hood, it uses a Support Vector Machine (SVM) classifier trained on the well-known MNIST dataset of handwritten digits. The application is built entirely in Python, using Tkinter for the graphical interface, Pillow for image handling, NumPy for numerical processing, and scikit-learn for the machine learning side of things.

What makes this more than just a "train a model and call it a day" project is the preprocessing pipeline. Getting a mouse-drawn digit to look anything like an MNIST training sample takes real effort — cropping, scaling, centering by center of mass, and normalizing — all before the model even sees the input. The system also supports two-digit recognition by segmenting the canvas into individual digit regions using vertical column profiling. The result is a responsive, self-contained tool that works offline, trains once, and saves the model for all future sessions.

---

## 1. Introduction

### Background

Handwritten digit recognition has been a cornerstone problem in machine learning and computer vision for decades. Long before neural networks became the go-to solution for everything, researchers were building systems that could read handwritten postal codes, process bank cheques, and digitize paper forms. The challenge sounds deceptively simple — a digit is just a shape, after all — but the variability in how different people write the same number makes it genuinely hard to get right.

The MNIST dataset, introduced by Yann LeCun and colleagues in 1998, became the standard benchmark for this problem. It contains 70,000 grayscale images of handwritten digits, each normalized to 28×28 pixels, collected from Census Bureau employees and high school students. For years, it was the go-to dataset for testing new ideas in classification, and it still holds up as a solid baseline today.

Early approaches to the problem leaned heavily on hand-crafted features — things like pixel histograms, edge detectors, or HOG descriptors — combined with classical classifiers like k-Nearest Neighbors or SVMs. These methods worked reasonably well but required a lot of domain knowledge to engineer the right features. The rise of deep learning, and CNNs in particular, changed the game by learning features directly from raw pixels. Today, state-of-the-art models exceed 99.7% accuracy on MNIST. But that doesn't mean classical methods are obsolete — SVMs in particular remain a strong, interpretable, and computationally lightweight option, especially when you don't have a GPU or don't want to pull in a heavy deep learning framework.

This project sits in that space: a practical, no-frills digit recognizer that runs on any Python-capable machine, trains in about a minute, and gets the job done.

### Importance of the Problem

It's easy to look at MNIST accuracy numbers and think the problem is solved. And in a narrow sense, it is — on clean, pre-processed test images, modern models are nearly perfect. But the moment you move from a curated dataset to real freehand input, things get messier.

When a user draws a digit with a mouse on a 300×300 canvas, the result looks nothing like an MNIST sample. The strokes are thicker, the proportions are off, the digit might be drawn in a corner or stretched across the whole canvas. If you just resize that to 28×28 and feed it to the model, you'll get poor results — not because the model is bad, but because the input doesn't match what the model was trained on.

This distribution mismatch is the core challenge this project addresses. Beyond single digits, recognizing two-digit numbers adds another layer of complexity: you need to figure out where one digit ends and the next begins, without any explicit annotation from the user. Getting that segmentation right — especially when strokes are close together or a digit has internal gaps — requires careful engineering.

These are the kinds of problems that don't show up in a Kaggle leaderboard but matter a lot when you're building something people actually use.

### Project Objectives

The goals of this project are straightforward but cover the full stack from data to deployment:

- Build a clean, interactive desktop GUI where users can draw digits naturally with a mouse.
- Train an SVM classifier on a meaningful subset of MNIST that achieves competitive accuracy without taking forever.
- Design a preprocessing pipeline that faithfully transforms freehand canvas input into MNIST-compatible 28×28 feature vectors.
- Implement vertical segmentation to support two-digit number recognition in a single drawing session.
- Persist the trained model to disk so the application loads instantly on every run after the first.
- Display prediction confidence so users get a sense of how certain the model is, not just what it guessed.

---

## 2. Data / Dataset Description

The dataset used in this project is MNIST (mnist_784), one of the most widely used benchmarks in machine learning history. It's fetched directly using scikit-learn's `fetch_openml` utility, which downloads and caches it automatically — no manual setup required.

**Dataset summary:**

| Property         | Value                          |
|------------------|-------------------------------|
| Total samples    | 70,000 (60k train / 10k test) |
| Subset used      | First 10,000 training samples |
| Image size       | 28 × 28 pixels (grayscale)    |
| Feature vector   | 784 flattened pixel values    |
| Classes          | 10 (digits 0–9)               |
| Pixel range      | 0–255, normalized to 0.0–1.0  |

Each image in MNIST is a centered, size-normalized grayscale digit on a black background. The centering is done by center of mass — meaning the "weight" of the ink is balanced around the middle of the 28×28 grid. This is a subtle but important detail that this project replicates during preprocessing, because it's what makes the drawn input actually look like the training data to the model.

The dataset is well-balanced across all 10 digit classes, so there's no class imbalance issue to worry about. The digits were written by a diverse group of people, which gives the dataset reasonable coverage of different handwriting styles.

For this project, only the first 10,000 samples are used for training. This is a deliberate trade-off: 10,000 samples is enough to train an SVM that generalizes well, while keeping training time under two minutes on a typical laptop CPU. Using the full 60,000 training samples would push training time to 10–20 minutes for an RBF SVM, which isn't practical for a first-run experience.

After training, the model is saved to `digit_svm.pkl` using `joblib`. On every subsequent launch, the app loads this file directly and skips training entirely — so the one-time cost is paid once and forgotten.

---

## 3. Methods / Models

### Model Overview

The core classifier is a Support Vector Machine with a Radial Basis Function (RBF) kernel. SVMs work by finding the hyperplane that best separates classes in a high-dimensional feature space, maximizing the margin between the closest points of each class (the support vectors). With the RBF kernel, the model can capture non-linear decision boundaries, which is essential for distinguishing visually similar digits like 3 and 8, or 6 and 9.

Here's a breakdown of every method and design choice in the system:

| Method                        | Description                                                                 | Notes / Justification                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| SVM (RBF kernel)              | Support Vector Machine with Radial Basis Function kernel                    | Strong non-linear classifier; well-suited for high-dimensional pixel feature spaces  |
| `C=5`                         | Regularization parameter controlling margin vs. misclassification trade-off | Higher C reduces training error; value of 5 balances bias-variance well on MNIST     |
| `gamma='scale'`               | Kernel coefficient set to `1 / (n_features × X.var())`                     | Automatically adapts to feature scale; more robust than a fixed gamma value          |
| `probability=True`            | Enables Platt scaling to output class probabilities                         | Required for confidence score display; adds slight training overhead                 |
| Gaussian Blur (radius=1)      | Applied to drawn image before segmentation and preprocessing                | Smooths jagged mouse strokes; reduces noise in column profile analysis               |
| Bounding Box Crop             | Crops to the tight bounding box of non-zero pixels                          | Removes empty canvas space; focuses the digit region for consistent scaling          |
| Aspect-Ratio Scaling (20×20)  | Scales digit to fit within a 20×20 box preserving aspect ratio              | Matches MNIST's internal 20×20 digit-within-28×28-canvas convention                 |
| Center-of-Mass Shift          | Shifts the digit so its center of mass aligns with pixel (14, 14)           | Directly replicates MNIST's centering method; critical for distribution alignment    |
| Single Normalization (÷255)   | Divides pixel values by 255 once after all spatial transforms               | Avoids double-normalization bug; keeps values in [0, 1] range expected by the model  |
| Vertical Segmentation         | Column profile analysis to detect gaps between digits                       | Enables two-digit recognition without a separate detection model                     |
| Gap Merging (< 5px)           | Merges column segments separated by fewer than 5 pixels                     | Handles broken strokes within a single digit without merging two separate digits     |

### Preprocessing Pipeline (per digit)

The preprocessing pipeline is the most critical part of the system. Every step exists for a specific reason, and skipping or getting any one of them wrong leads to noticeably worse predictions.

```
Raw canvas (300×300, grayscale)
        │
        ▼
Gaussian Blur (radius=1)
  → Smooths jagged mouse strokes
        │
        ▼
Threshold (pixel > 10) → binary mask
  → Separates ink from background noise
        │
        ▼
Bounding box crop (row/col extents)
  → Removes empty space around the digit
        │
        ▼
Scale to fit 20×20 (preserve aspect ratio, LANCZOS resampling)
  → Matches MNIST's digit size convention
        │
        ▼
Place centered in 28×28 zero canvas
  → Adds the 4px border MNIST uses around the 20×20 region
        │
        ▼
Shift by center-of-mass offset → align center of mass to (14, 14)
  → Replicates MNIST's centering normalization exactly
        │
        ▼
Normalize: divide by 255.0 (single step)
  → Converts to float32 in [0.0, 1.0]
        │
        ▼
Flatten to 784-dim vector → SVM input
```

### Segmentation Pipeline (for 2-digit input)

When the user draws two digits, the system needs to figure out where one ends and the other begins. Rather than using a complex object detection approach, the system uses a simple but effective column profile analysis — essentially asking "which vertical columns of the image have any ink in them?"

```
Blurred image array
        │
        ▼
Row extent: find topmost and bottommost non-zero rows
  → Defines the vertical bounds of the drawing
        │
        ▼
Column profile: count non-zero pixels per column (within row extent)
  → Produces a 1D signal showing ink density across the width
        │
        ▼
Detect contiguous non-zero column spans → raw segments
  → Each span is a candidate digit region
        │
        ▼
Merge segments with gap < 5px
  → Handles broken strokes within a single digit (e.g., a dotted "i" or split "7")
        │
        ▼
If > 2 segments: merge pair with smallest gap until ≤ 2
  → Caps at 2 digits; merges the most ambiguous boundary first
        │
        ▼
Crop each segment → individual digit arrays
        │
        ▼
Predict each independently → concatenate results left to right
```

---

## 4. Experimental Setup

Getting the environment right matters as much as the algorithm. Here's exactly how the system is configured, so the results are reproducible.

**Software environment:**

| Component     | Detail                              |
|---------------|-------------------------------------|
| Language      | Python 3.x                          |
| GUI           | Tkinter (standard library)          |
| Image I/O     | Pillow (PIL)                        |
| Numerics      | NumPy                               |
| ML Framework  | scikit-learn                        |
| Model Storage | joblib                              |
| Platform      | Windows (win32)                     |

**Training configuration:**

The model is trained once on first launch and saved to disk. Training happens in a background daemon thread so the UI stays fully responsive — you can see the canvas and buttons while the model trains in the background.

- Dataset source: `fetch_openml('mnist_784', version=1)`
- Training subset: first 10,000 samples
- Pixel normalization: divide by 255.0 → float32 in [0, 1]
- Classifier: `sklearn.svm.SVC(kernel='rbf', C=5, gamma='scale', probability=True)`
- Persistence: `joblib.dump(clf, 'digit_svm.pkl')` after training; `joblib.load()` on subsequent runs

**Canvas and drawing configuration:**

The canvas is intentionally large (300×300) to give users comfortable drawing space. The brush is a filled ellipse with radius 10px, which produces strokes thick enough to be visible but not so thick they bleed into adjacent digit regions.

- Canvas size: 300 × 300 pixels
- Brush radius: 10px (filled ellipse)
- Stroke interpolation: linear interpolation between consecutive drag events, stepping every 3px, to fill gaps when the mouse moves quickly
- Canvas background: `#181825` (near-black); strokes: white (pixel value 255)
- Internal PIL image: grayscale `"L"` mode, same 300×300 size, kept in sync with the Tkinter canvas

**Inference flow:**

When the user clicks "Predict", the system runs the full pipeline synchronously on the main thread (fast enough to feel instant):

1. The PIL image is passed to `segment_digits()`, which returns a list of 1 or 2 cropped arrays.
2. Each crop goes through `preprocess_crop()` to produce a 784-dim float32 vector.
3. `model.predict_proba()` returns a 10-class probability distribution.
4. `argmax` picks the predicted digit; the probability at that index becomes the confidence score.
5. For two digits, both predictions and both confidence scores are displayed side by side.

---

## 5. Results

### Results Screenshots

> _Add screenshots of the running application here. Suggested captures:_
> - Single digit prediction — e.g., drawing "7" → predicted "7" at 94.2% confidence
> - Two-digit prediction — e.g., drawing "4" and "2" with a gap → predicted "42" at 91.3% | 88.7%
> - A previously failing case — digit "6" now correctly predicted after the preprocessing fix
> - Status bar showing "Found 2 digit(s) — Ready" after segmentation detects two regions
> - A low-confidence case — e.g., a messy "9" predicted as "9" at 61% — showing the confidence score working as intended

### Discussion

The SVM with RBF kernel trained on 10,000 MNIST samples achieves approximately **94–96% accuracy** on the MNIST test set at this scale. That's a solid result for a classical model on a subset of the data, and it holds up well in practice for cleanly drawn digits.

That said, real-world freehand input is messier than the test set, and a few patterns consistently cause trouble:

**Stroke thickness** is the biggest source of error. The 10px brush produces strokes that are noticeably thicker than typical MNIST digits. The 20×20 scaling step compresses the digit proportionally, which helps, but very thick strokes can cause the model to see a "filled blob" rather than a recognizable shape — especially for digits like 0, 6, and 8 that have enclosed regions.

**Drawing style variation** is unavoidable. Some people write their 1s with a serif base, their 7s with a crossbar, or their 4s as open triangles. The model has seen a range of styles in its 10,000 training samples, but it can still be thrown off by unusual handwriting.

**Segmentation sensitivity** is the trickiest part of two-digit recognition. The 5px gap threshold is a deliberate balance — tight enough to preserve the gap between two separate digits, but loose enough to merge broken strokes within a single digit. If the user draws two digits very close together (less than 5px apart), the system will treat them as one. If a digit like "7" has a gap in its stroke, the system might split it into two segments. Users quickly learn to leave a clear gap between digits, which makes the system work reliably.

The most consistently problematic digits are **6, 8, and 9**. They share similar circular structures, and the model sometimes confuses them — particularly 6 and 9, which are rotations of each other. The center-of-mass alignment step helps significantly here, because it ensures the ink is always positioned consistently relative to the 28×28 grid, regardless of where on the canvas the user drew.

One important bug that was caught and fixed during development: the preprocessing pipeline was accidentally dividing by 255 twice — once implicitly through PIL's float conversion and once explicitly at the end. This caused all pixel values to be near zero, which made the model see nearly blank images. Fixing this to a single normalization step at the end produced a noticeable improvement in prediction quality.

---

## 6. Comparison & Analysis

It's worth stepping back and asking: why SVM? And how does it compare to the alternatives that were considered?

| Approach                           | Accuracy (10k subset) | Training Time  | Inference Speed | Real-time Usability | Notes                                          |
|------------------------------------|----------------------|----------------|-----------------|---------------------|------------------------------------------------|
| SVM RBF (C=5, gamma='scale')       | ~94–96%              | ~60–90 sec     | Fast            | Yes (after load)    | Current implementation; best accuracy here     |
| SVM RBF (C=1, gamma=0.05)          | ~91–93%              | ~45 sec        | Fast            | Yes                 | Earlier version; weaker on 6/8/9 confusion     |
| LinearSVC + CalibratedClassifierCV | ~92–94%              | ~10–15 sec     | Very fast       | Yes                 | Much faster to train; slightly lower accuracy  |
| k-NN (k=3)                         | ~96%                 | None (lazy)    | Slow            | Borderline          | High memory usage; slow at inference on 10k    |
| CNN (LeNet-5 style)                | ~99%+                | Minutes (GPU)  | Very fast       | Yes                 | Best accuracy; requires PyTorch or TensorFlow  |
| Random Forest (100 trees)          | ~93–95%              | ~30 sec        | Fast            | Yes                 | Comparable accuracy; less interpretable        |

**Why not CNN?** CNNs are objectively better at this task. But they require either PyTorch or TensorFlow, which are large dependencies that feel like overkill for a lightweight desktop tool. They also take longer to train from scratch and are harder to inspect. For a project that prioritizes simplicity and portability, SVM is the right call.

**Why not k-NN?** k-NN actually achieves competitive accuracy on MNIST, but its inference time scales with the size of the training set. With 10,000 samples and 784 features, each prediction requires computing distances to all 10,000 training points — which is noticeably slow in a real-time interactive setting.

**Why not LinearSVC?** LinearSVC trains dramatically faster (10–15 seconds vs. 60–90 seconds) and is a reasonable alternative. It was actually used in an intermediate version of this project. The trade-off is that it requires `CalibratedClassifierCV` to produce probabilities (since LinearSVC doesn't natively support them), and the calibration quality is slightly lower than Platt scaling on the RBF SVM. For a tool where confidence scores are a key part of the UX, the RBF SVM's better-calibrated probabilities are worth the extra training time.

**The gamma fix** deserves a mention. The original implementation used `gamma=0.05`, a fixed value that worked reasonably well but wasn't tuned to the data. Switching to `gamma='scale'` — which automatically computes the kernel coefficient based on the feature variance — improved accuracy on visually similar digits (6, 8, 9) without any manual tuning.

---

## 7. Conclusion

This project started with a simple question — can you build a digit recognizer that actually works when you draw with a mouse? — and the answer turned out to be yes, but only if you take the preprocessing seriously.

The model itself is not the hard part. An SVM trained on MNIST is a well-understood, well-documented approach. The hard part is making the input look like the training data. Every step in the preprocessing pipeline — the blur, the crop, the aspect-ratio scaling, the center-of-mass shift, the single normalization — exists because without it, the model sees something it wasn't trained on and makes poor predictions. Getting that pipeline right is what separates a demo that works in a notebook from an application that works in someone's hands.

The two-digit segmentation adds another layer of practical complexity. Column profile analysis is a simple idea, but the details matter: the gap threshold, the merging logic, the cap at two segments. These aren't arbitrary numbers — they're the result of testing with real drawn input and tuning until the behavior felt natural.

A few things stand out as lessons from this project:

- Distribution alignment between training data and real input is often more impactful than model choice.
- Model persistence is a small engineering detail that makes a big difference in usability — nobody wants to wait a minute every time they open an app.
- Confidence scores add real value. Knowing the model is 61% confident on a prediction tells you something useful that a bare digit label doesn't.
- The system is designed to be modular. Swapping the SVM for a CNN would only require changes to `train_model()` and `preprocess_crop()`. The UI, the drawing logic, and the segmentation pipeline would all stay exactly the same.

Looking ahead, there are several natural directions to extend this work. Supporting more than two digits would require a more robust segmentation approach — possibly connected component labeling rather than column profiling. Adding a confidence threshold (e.g., showing "?" when confidence drops below 50%) would make the system more honest about its uncertainty. And retraining with augmented data — thicker strokes, slight rotations, varying scales — would help close the gap between MNIST's clean samples and the messier reality of mouse-drawn input.

For now, though, it does what it set out to do: you draw a digit, it tells you what it is. That's a satisfying thing to build.

---

## 8. References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
   > The original MNIST paper. Introduced the dataset and the LeNet architecture, and established the benchmark that the field has used ever since.

2. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
   > The foundational paper on SVMs. Introduced the concept of maximum-margin classifiers and the kernel trick that makes non-linear classification tractable.

3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
   > The paper behind the scikit-learn library used for SVM training, model persistence, and dataset loading in this project.

4. MNIST Database — http://yann.lecun.com/exdb/mnist/
   > The official MNIST homepage, with the original dataset files and a historical leaderboard of methods and their accuracies.

5. OpenML MNIST-784 — https://www.openml.org/d/554
   > The OpenML version of MNIST used via `fetch_openml` in scikit-learn. Provides the same data in a convenient API-accessible format.

6. Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 10(3), 61–74.
   > Describes Platt scaling, the technique used by scikit-learn's `SVC(probability=True)` to convert SVM decision scores into calibrated class probabilities.

7. Chang, C.-C., & Lin, C.-J. (2011). LIBSVM: A library for support vector machines. *ACM Transactions on Intelligent Systems and Technology*, 2(3), 1–27.
   > The underlying LIBSVM library that scikit-learn's SVC wraps. Understanding its internals helps explain training time behavior at different dataset sizes.

8. Python Software Foundation. Tkinter — Python interface to Tcl/Tk.
   https://docs.python.org/3/library/tkinter.html
   > Official documentation for the Tkinter GUI toolkit used to build the application's canvas, buttons, and labels.

9. Clark, A. (2015). Pillow (PIL Fork) Documentation.
   https://pillow.readthedocs.io
   > Documentation for the Pillow library used for image creation, drawing, Gaussian blur, and LANCZOS resampling in the preprocessing pipeline.

10. NumPy Development Team. NumPy Documentation.
    https://numpy.org/doc/
    > NumPy powers all the array operations in preprocessing — bounding box detection, center-of-mass calculation, canvas construction, and feature vector flattening.
