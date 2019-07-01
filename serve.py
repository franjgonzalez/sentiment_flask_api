import pickle

import tensorflow as tf
from tensorflow.contrib import predictor

from model.attention_text_classification.src import main
from model.attention_text_classification.src.utils import utils
from model.attention_text_classification.src.utils import data_utils


def process_input_data(input_data, tokenizer):
    """Process input data for model to use."""
    # Clean raw text
    text = data_utils.text_to_wordlist(input_data)

    # Tokenize and pad text
    padded_token_seq = data_utils.get_padded_sequences(tokenizer, [text])

    return padded_token_seq


def process_output_data(predictions, padded_token_seq, tokenizer):
    """Process output data."""
    # Regenerate text
    text = [tokenizer.index_word[idx] for idx in padded_token_seq[0] if idx != 0]

    # Normalize alphas for highlighting text
    mask = padded_token_seq[0] != 0
    alphas = (predictions["alphas"][0] / predictions["alphas"][0].max())[mask]

    output_data = {
        "class": predictions["class_ids"][0],
        "probability": predictions["probabilities"][0],
        "text": text,
        "alphas": alphas,
    }
    return output_data


def get_model_api():
    """Returns lambda function for API.

    Returns:
        type: Description of returned object.

    """
    # Load tokenizer
    with open("data/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Initilize predictor from saved model
    predict_fn = predictor.from_saved_model("model/saved_model")

    def model_api(input_data):
        """Model API.

        Args:
            input_data: submitted to the API, raw string

        Returns:
            output_data

        """
        # 2. Process input data
        padded_token_seq = process_input_data(input_data, tokenizer)

        # 3. Call model predict function
        predictions = predict_fn({"feature": padded_token_seq})

        # 4. Process the output
        output_data = process_output_data(predictions, padded_token_seq, tokenizer)

        # Return data
        return output_data

    return model_api
