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
    text_list = [tokenizer.index_word[idx] for idx in padded_token_seq[0] if idx != 0]
    text = " ".join(text_list)

    # Normalize alphas for highlighting text
    mask = padded_token_seq[0] != 0
    alphas = (predictions["alphas"][0] / predictions["alphas"][0].max())[mask]

    # HTML attention map
    attn_text = visualize_attention(
        class_id=int(predictions["class_ids"][0]), text_list=text_list, alphas=alphas
    )

    output_data = {
        "class_id": int(predictions["class_ids"][0]),
        "probability": float(predictions["probabilities"][0]),
        "text": text,
        "attn_text": attn_text,
    }
    return output_data


def visualize_attention(class_id, text_list, alphas):
    """Create html string with highlighted words"""
    # Choose color based on predicted class
    if class_id == 1:
        r, g, b = 0, 255, 0
    else:
        r, g, b = 255, 0, 0

    html_list = []
    for word, alpha in zip(text_list, alphas):
        html_list.append(
            f"<font style='background: rgba({r}, {g}, {b}, {alpha})'>{word}</font>"
        )

    return " ".join(html_list)


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
