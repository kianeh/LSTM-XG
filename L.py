# Import necessary libraries
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from datetime import datetime
import os

if __name__ == "__main__":
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    
    # Load and preprocess the dataset
    df = pd.read_csv(r"C:\Users\kiane\BankChurners.csv").dropna().drop('CLIENTNUM', axis=1)
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    for col in categorical_cols:
        df[col], _ = pd.factorize(df[col])
    X = df.drop(['Attrition_Flag'], axis=1)
    y = df['Attrition_Flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    # Define the model architecture
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.5),
        LSTM(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Train the model
    history = model.fit(X_train, y_train, batch_size=128, epochs=50,
                        validation_data=(X_test, y_test), class_weight=class_weight_dict, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Mean Validation Loss:", loss)
    print("Mean Validation Accuracy:", accuracy)

    # Plot training & validation accuracy and loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    # Save the plot
    plot_path = os.path.join('cache', f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_history.png')
    fig.savefig(plot_path)
    print(f"Saved history plot at: {plot_path}")
