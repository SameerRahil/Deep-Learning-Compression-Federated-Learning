# Network Compression and Federated Learning

Project 6 of CS 437 Deep Learning, LUMS, Spring 2026. The first notebook makes a ResNet smaller with pruning and distillation; the second builds federated learning from scratch and takes it to a real dataset.

## Parts

| Part | Notebook | What it covers |
| --- | --- | --- |
| 1. Pruning and knowledge distillation | [`1_pruning_and_distillation.ipynb`](notebooks/1_pruning_and_distillation.ipynb) ([nbviewer](https://nbviewer.org/github/SameerRahil/Deep-Learning-Compression-Federated-Learning/blob/main/notebooks/1_pruning_and_distillation.ipynb)) | Unstructured and structured (channel-wise) pruning of a ResNet-34 on CIFAR-100 with per-layer sensitivity sweeps, Grad-CAM comparisons, Hinton logit-matching distillation, FitNets hint distillation, whether a larger teacher helps, and a lab that combines pruning and distillation under a fixed budget. |
| 2. Federated learning | [`2_federated_learning.ipynb`](notebooks/2_federated_learning.ipynb) ([nbviewer](https://nbviewer.org/github/SameerRahil/Deep-Learning-Compression-Federated-Learning/blob/main/notebooks/2_federated_learning.ipynb)) | A federated pipeline from scratch (IID, Dirichlet and quantity-skew splits over 10 clients), a check that one-epoch FedAvg equals centralized gradient descent, a FedProx sweep, client drift and training diagnostics, distracted-driver detection on State Farm (SFD3), and a fairness-aware aggregation rule. |

If GitHub cannot display a notebook, the nbviewer link next to it opens the same file.

## Results

**Compression (CIFAR-100, ResNet-34 baseline 60.45%, 21.3M parameters, 85.5 MB):**

| Model | Accuracy | Parameters | Size |
| --- | --- | --- | --- |
| 70% unstructured pruning | 60.55% | 6.4M nonzero | dense storage, no speedup |
| 70% structured pruning, fine-tuned | 47.70% | 6.4M | 25.8 MB, 4.16 ms |
| ResNet-18, trained alone | 59.97% | 11.2M | 45.0 MB |
| ResNet-18, logit distillation | 59.88% | 11.2M | 45.0 MB |
| ResNet-18, best distillation setting (T = 16, alpha = 0.1) | 61.09% | 11.2M | 45.0 MB |
| Pruning + distillation combined | 57.63% | 6.7M | 27.0 MB, 2.83 ms |

Structured pruning collapsed to 1.16% before fine-tuning, and its Grad-CAM maps no longer matched the baseline (SSIM 0.27, against 0.9997 for unstructured pruning).

**Federated learning:**

- One-epoch FedAvg matched centralized gradient descent to within 1.3e-15 over 50 rounds.
- FedProx on Dirichlet-skewed CIFAR-10: final accuracy fell from 56.9% (mu = 0) to 38.1% (mu = 10).
- Distracted-driver detection (pretrained ResNet-18): centralized 81.1%, FedAvg 83.0% (worst driver 62.9%), FedProx 79.8% (worst driver 51.4%).
- Fairness-aware aggregation raised worst-client accuracy from 43.8% to 46.3% and cut the variance of client accuracy from 0.0057 to 0.0020, passing all three fairness checks.

## Stack

Python, PyTorch, torchvision, pytorch-grad-cam, pandas, Matplotlib.

## Running the notebooks

The notebooks were run on Google Colab or Kaggle with a GPU. Open one in Colab, set the runtime to GPU, and run the cells in order. CIFAR-100 and CIFAR-10 download through torchvision. The SFD3 part uses the State Farm Distracted Driver Detection data from Kaggle, which needs a Kaggle account and the competition rules accepted.

## About

Coursework for CS 437 Deep Learning, LUMS, Spring 2026, shared as part of my portfolio (course grade: A+). The notebooks keep the original handout text and my implementations, answers and outputs.

If you are taking this course, please do your own work.

**Sameer Rahil**, BS Computer Science, LUMS. More projects at [github.com/SameerRahil](https://github.com/SameerRahil).
