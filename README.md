# CIFAR-10 Object Recognition using ResNet50

This project implements an image classification pipeline on the CIFAR-10 dataset using the **ResNet50** deep convolutional neural network. The model is trained to recognize objects from 10 distinct categories with transfer learning and fine-tuning techniques.

## 📌 Project Overview

- **Dataset**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Task**: Multi-class image classification
- **Model**: Transfer Learning using ResNet50
- **Framework**: TensorFlow & Keras

## 🧰 Key Steps

1. **Data Preprocessing**  
   - Loaded CIFAR-10 dataset from `keras.datasets`
   - Normalized image pixel values
   - Converted categorical labels to one-hot encoding

2. **Model Architecture**  
   - Used `ResNet50` with pre-trained ImageNet weights (`include_top=False`)
   - Added GlobalAveragePooling, Dense, Dropout, and final classification layer
   - Applied softmax activation for multi-class output

3. **Training Configuration**  
   - Optimizer: Adam  
   - Loss: Categorical Crossentropy  
   - Metrics: Accuracy  
   - Trained for 10 epochs (modifiable)

4. **Evaluation**  
   - Assessed model performance on test set  
   - Plotted training/validation accuracy and loss  
   - Showcased sample predictions for visualization

## 📊 Results

- The model effectively learns to distinguish between the 10 CIFAR-10 classes.
- ResNet50’s deep architecture improves accuracy and generalization compared to simpler CNNs.

## 🚀 How to Run

1. Clone this repository or open the notebook directly in Google Colab.
2. Run all the cells in order to train and evaluate the model.
3. CIFAR-10 dataset is automatically loaded via Keras; no extra download needed.

## 📁 File

- `CIFAR_10_Object_Recognition_using_ResNet50.ipynb` – Main Jupyter notebook containing the complete project code.

## 🧠 Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  

## 📌 Future Improvements

- Use data augmentation for better generalization  
- Experiment with other architectures (e.g., EfficientNet, MobileNet)  
- Perform hyperparameter tuning and early stopping  
- Save and deploy the trained model as a web app

---
