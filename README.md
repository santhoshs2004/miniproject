# miniproject

                                     Human Activity Recognition using Multiheaded CNN & LSTM
This project aims to build a Human Activity Recognition (HAR) model using a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The model is designed to classify human activities based on time-series sensor data, specifically from accelerometers and gyroscopes.

Project Overview
Human Activity Recognition (HAR) is an important application in various fields, including healthcare, sports, and robotics. In this project, we use a multiheaded CNN to extract spatial features from different sensor data streams and an LSTM layer to capture the temporal dependencies in the data.

Key Components
Multiheaded CNN: Each sensor type (e.g., accelerometer, gyroscope) has its own CNN branch for spatial feature extraction.
LSTM Layer: Combines the CNN-extracted features to capture temporal patterns.
Classification Layer: A dense layer with softmax activation is used for activity classification.
Dataset
The model can be trained on any HAR dataset, such as the UCI HAR Dataset or other time-series sensor data. For this project:

The input consists of time-series sensor data (e.g., accelerometer, gyroscope).
The output is the predicted activity (e.g., walking, sitting, running).
Data Preprocessing
Normalize the data to scale the sensor readings.
Reshape the data into the required input format for CNN and LSTM (e.g., (batch_size, time_steps, features)).
Split the data into training and testing sets.
Model Architecture
The model consists of the following main components:

Multiheaded CNN:

Separate CNN branches for each sensor modality.
Convolutional layers extract spatial features from the sensor data.
MaxPooling and Flatten layers for dimensionality reduction.
LSTM Layer:

The features from the CNN branches are concatenated and passed into an LSTM layer.
The LSTM captures the temporal dependencies in the data.
Dense Output Layer:

A fully connected layer with softmax activation for classification.
Model Code Example
python
Copy code
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Input, Concatenate
from tensorflow.keras.models import Model

def create_cnn_branch(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    return input_layer, x

# Create CNN branches for accelerometer and gyroscope data
input_acc, branch_acc = create_cnn_branch((128, 3))  # Accelerometer
input_gyro, branch_gyro = create_cnn_branch((128, 3))  # Gyroscope

# Concatenate and pass through LSTM
combined = Concatenate()([branch_acc, branch_gyro])
lstm_output = LSTM(100)(combined)

# Dense output layer for classification
output = Dense(num_classes, activation='softmax')(lstm_output)

# Build and compile the model
model = Model(inputs=[input_acc, input_gyro], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([train_acc_data, train_gyro_data], train_labels, epochs=10, batch_size=64, validation_data=([test_acc_data, test_gyro_data], test_labels))
How to Run the Project
Clone the Repository:
Clone this project from the repository:

bash
Copy code
git clone https://github.com/yourusername/har-multiheaded-cnn-lstm.git
Install Dependencies:
Install the required Python libraries by running:

bash
Copy code
pip install -r requirements.txt
Prepare the Dataset:

Download and preprocess the HAR dataset (such as UCI HAR Dataset).
Split the data into training and test sets.
Train the Model:
Train the model using the following command:

bash
Copy code
python train.py
Evaluate the Model:
After training, evaluate the model on the test set:

bash
Copy code
python evaluate.py
Requirements
Python 3.x
TensorFlow 2.x
Numpy
Pandas
Matplotlib
Install the dependencies using:

bash
Copy code
pip install tensorflow numpy pandas matplotlib
Results
The model's performance is evaluated based on accuracy, precision, recall, and F1-score.
You can visualize the training history (accuracy/loss over epochs) using Matplotlib.
