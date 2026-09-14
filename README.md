# ✍️ Hindi Digit Classifier

A compact convolutional neural network (CNN) that recognizes handwritten Hindi (Devanagari) digits 0–9. Built for the *Introduction to Image & Video Processing* course (KEN3238) at Maastricht University.

We chose a small CNN because the task is image-based, the labels are balanced and clean, and a CNN learns local stroke patterns directly from the pixels.

## Approach

- **Data:** 17,000 labelled training images and 3,000 test images, with a stratified 85/15 train/validation split
- **Model:** 7 convolutional blocks (Conv → BatchNorm → ReLU, 24 → 128 channels) with max-pooling and spatial dropout, followed by global average pooling and a small fully connected head
- **Training:** AdamW, a OneCycle learning-rate schedule, label smoothing and data augmentation (random rotation, scaling and shifts); the best epoch is checkpointed by validation accuracy
- **Inference:** test-time augmentation that averages predictions over 9 shifted copies of each image, optionally across several checkpoints
- **Hardware:** NVIDIA GPUs (CUDA with mixed precision), Apple silicon (MPS) or CPU, detected automatically

## Run it

```bash
pip install torch torchvision numpy pandas pillow
python train_cnn.py                   # train, save best_cnn.pt, write submission.csv
python train_cnn.py --skip-training   # predict with the included checkpoint
```

Useful flags: `--epochs` (default 30), `--batch-size` (256), `--lr` (0.003), `--device auto|cuda|mps|cpu`.

## Repository layout

```
train_cnn.py        Data loading, model, training loop and prediction
train/, train.csv   Training images (one folder per digit) and labels
test/, test.csv     Test images and IDs
best_cnn.pt         Trained model checkpoint
submission.csv      Predictions for the test set
```

## Team

Antonia Maria Constantin · Toma Cristian Nitu · Vlad Stoica
