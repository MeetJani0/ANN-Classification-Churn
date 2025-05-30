# ANN-Classification-Churn
This project implements an Artificial Neural Network (ANN) to predict customer churn based on various customer attributes. By analyzing features such as credit score, geography, gender, age, tenure, balance, and more, the model aims to identify customers who are likely to leave a bank.([GitHub][1], [GitHub][2])

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Evaluation Metrics](#evaluation-metrics)
* [Results](#results)
* [License](#license)

## Overview

Customer churn prediction is vital for businesses to retain clients and maintain revenue. This project utilizes a deep learning approach to classify whether a customer will exit the bank. The ANN model is trained on a dataset containing various customer features and their churn status.([GitHub][1], [GitHub][3])

## Dataset

The dataset used is the [Churn Modelling dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling), which includes the following features:([Kaggle][4])

* RowNumber
* CustomerId
* Surname
* CreditScore
* Geography
* Gender
* Age
* Tenure
* Balance
* NumOfProducts
* HasCrCard
* IsActiveMember
* EstimatedSalary
* Exited (Target Variable)

The dataset is included in the repository as `Churn_Modelling.csv`.([GitHub][2])

## Project Structure

The repository contains the following files:

* `app.py`: Script to run the application.
* `experiments.ipynb`: Jupyter notebook containing exploratory data analysis and model experimentation.
* `prediction.ipynb`: Notebook for making predictions using the trained model.
* `model.h5`: Saved trained ANN model.
* `label_encode_gender.pkl`: Pickle file for label encoding the 'Gender' feature.
* `onehot_encoder_geo.pkl`: Pickle file for one-hot encoding the 'Geography' feature.
* `scaler.pkl`: Pickle file for feature scaling.
* `requirements.txt`: List of required Python packages.
* `README.md`: Project documentation.([GitHub][5], [GitHub][6], [Medium][7])

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/MeetJani0/ANN-Classification-Churn.git
   cd ANN-Classification-Churn
   ```



2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```



3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```



## Usage

1. **Run the application:**

   ```bash
   python app.py
   ```



This will start the application, allowing you to input customer data and receive churn predictions.

2. **Using the Jupyter notebooks:**

   * `experiments.ipynb`: Contains data exploration, preprocessing steps, model training, and evaluation.
   * `prediction.ipynb`: Demonstrates how to load the trained model and make predictions on new data.([GitHub][6])

## Model Architecture

The ANN model is built using TensorFlow and Keras. The architecture includes:([GitHub][1], [Medium][7])

* **Input Layer**: Accepts preprocessed features.
* **Hidden Layers**: Two dense layers with ReLU activation functions.
* **Output Layer**: Single neuron with a sigmoid activation function for binary classification.([GitHub][2], [Kaggle][4])

## Evaluation Metrics

The model's performance is evaluated using the following metrics:

* **Accuracy**: Proportion of correctly predicted instances.
* **Precision**: Proportion of positive identifications that were actually correct.
* **Recall**: Proportion of actual positives that were correctly identified.
* **F1-Score**: Harmonic mean of precision and recall.
* **Confusion Matrix**: Table to describe the performance of the classification model.

## Results

The trained ANN model achieved the following performance on the test dataset:

* **Accuracy**: 84.15%
* **Precision**: \[Insert Precision]
* **Recall**: \[Insert Recall]
* **F1-Score**: \[Insert F1-Score]\([GitHub][6], [GitHub][5])

*Note: Replace \[Insert Precision], \[Insert Recall], and \[Insert F1-Score] with actual values obtained from your model evaluation.*

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

Feel free to customize this README further to match any additional features or changes in your project. Let me know if you need assistance with anything else!

[1]: https://github.com/sanskaryo/Churn-Prediction-Using_ANN?utm_source=chatgpt.com "Banking customer churn prediction using ann - GitHub"
[2]: https://github.com/Jayita11/ANN-Classification-Customer-Churn-Prediction?utm_source=chatgpt.com "Jayita11/ANN-Classification-Customer-Churn-Prediction - GitHub"
[3]: https://github.com/Gulshank0719/Churn--Deep-Learning-ANN?utm_source=chatgpt.com "Gulshank0719/Churn--Deep-Learning-ANN - GitHub"
[4]: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling?utm_source=chatgpt.com "Churn Modelling - Kaggle"
[5]: https://github.com/topics/customer-churn-prediction?utm_source=chatgpt.com "customer-churn-prediction · GitHub Topics"
[6]: https://github.com/vinit714/ANN-Classification-model-to-predict-the-Customer-Churn?utm_source=chatgpt.com "vinit714/ANN-Classification-model-to-predict-the-Customer-Churn"
[7]: https://parisrohan.medium.com/bank-customer-churn-prediction-using-ann-6499bf805b6?utm_source=chatgpt.com "Bank customer churn prediction using ANN | by Rohan Paris - Medium"
