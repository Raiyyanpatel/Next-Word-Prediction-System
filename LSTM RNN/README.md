# Next Word Prediction with LSTM

A deep learning project that uses LSTM (Long Short-Term Memory) neural networks to predict the next word in a sequence. The model is trained on Shakespeare's Hamlet and features a user-friendly Streamlit web interface.

## Overview

This project implements a next-word prediction system using LSTM networks with early stopping. Users can input a sequence of words, and the model predicts the most likely next word based on patterns learned from the training text.

## Features

- **LSTM Neural Network**: Trained on Shakespeare's Hamlet text
- **Early Stopping**: Prevents overfitting during training
- **Interactive Web Interface**: Built with Streamlit for easy interaction
- **Real-time Predictions**: Instant next-word suggestions

## Project Structure

```
LSTM RNN/
├── app.py                                      # Streamlit web application
├── experiemnts.ipynb                           # Model training and experiments
├── hamlet.txt                                  # Training data (Shakespeare's Hamlet)
├── next_word_lstm.h5                          # Trained LSTM model
├── next_word_lstm_model_with_early_stopping.h5 # Model with early stopping
├── tokenizer.pickle                            # Tokenizer for text preprocessing
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

## Requirements

- Python 3.8+
- TensorFlow 2.15.0
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- TensorBoard
- SciKeras

## Installation

1. Clone or download this repository

2. Navigate to the project directory:
```bash
cd "LSTM RNN"
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

1. Ensure you have the trained model (`next_word_lstm.h5`) and tokenizer (`tokenizer.pickle`) in the project directory

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the provided local URL (typically `http://localhost:8501`)

4. Enter a sequence of words in the text input field (e.g., "To be or not to")

5. Click "Predict Next Word" to see the model's prediction

### Training the Model

To train or experiment with the model:

1. Open `experiemnts.ipynb` in Jupyter Notebook or JupyterLab
2. Run the cells to preprocess the data, train the model, and evaluate performance
3. The trained model will be saved as an `.h5` file

## How It Works

1. **Text Preprocessing**: The input text is tokenized using the pre-trained tokenizer
2. **Sequence Preparation**: The tokenized sequence is padded to match the model's expected input length
3. **Prediction**: The LSTM model processes the sequence and outputs probability distributions over the vocabulary
4. **Word Selection**: The word with the highest probability is selected as the prediction

## Model Architecture

The LSTM model is trained with:
- Sequential architecture with LSTM layers
- Early stopping to prevent overfitting
- Categorical cross-entropy loss
- Adam optimizer

## Training Data

The model is trained on Shakespeare's "Hamlet" (`hamlet.txt`), providing it with classic English literature patterns for word prediction.

## Future Improvements

- Support for multiple training texts
- Model fine-tuning options
- Beam search for multiple word predictions
- Temperature-based sampling for more creative predictions
- Export predictions history

## License

This project is for educational purposes.

## Acknowledgments

- Training data: William Shakespeare's Hamlet
- Framework: TensorFlow/Keras
- Web Interface: Streamlit
