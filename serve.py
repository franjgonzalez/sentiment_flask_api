import pickle

import tensorflow as tf

from model.src import main
from model.src.utils import utils
from model.src.utils import data_utils


def process_input_data(input_data, tokenizer):
    """Process input data for model to use."""
    # Clean raw text
    text = data_utils.text_to_wordlist(input_data)

    # Tokenize and pad text
    padded_token_seq = data_utils.get_padded_sequences(tokenizer, [text])

    return padded_token_seq


def process_output_data(pred_gen, padded_token_seq, tokenizer):
    """Process output data."""
    # Yield prediction dictionary from generator
    pred_dict = next(pred_gen)

    # Regenerate text
    text = [tokenizer.index_word[idx] for idx in padded_token_seq[0] if idx != 0]

    # Normalize alphas for highlighting text
    mask = padded_token_seq[0] != 0
    alphas = (pred_dict["alphas"] / pred_dict["alphas"].max())[mask]

    output_data = {
        "class": pred_dict["class_ids"][0],
        "probability": pred_dict["probabilities"][0],
        "text": text,
        "alphas": alphas,
    }
    return output_data


def get_model_api():
    """Returns lambda function for API.

    Returns:
        type: Description of returned object.

    """
    # Load hyperparameters and tokenizer
    with open("data/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    params = utils.load_config("src/config.yml")
    params.tokenizer = tokenizer

    # 1. Initialize model.
    estimator = tf.estimator.Estimator(
        model_fn=main.model_fn, params=params, model_dir="data/ckpt/"
    )

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
        pred_gen = estimator.predict(
            input_fn=lambda: data_utils.input_fn(
                features=padded_token_seq, labels=None, batch_size=1, buffer_size=1
            )
        )

        # 4. Process the output
        output_data = process_output_data(pred_gen, padded_token_seq, tokenizer)

        # Return data
        return output_data

    return model_api
