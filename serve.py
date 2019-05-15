# Import model


def get_model_api():
    """Returns lambda function for API.

    Returns:
        type: Description of returned object.

    """

    # Initialize model.
    # This could be either a pb file or restoration from checkpoints
    #
    # model = model.Model()

    def model_api(input_data):
        """Model API.

        Args:
            input_data (type): Description of parameter `input_data`.

        Returns:
            type: Description of returned object.

        """
        # Process input data
        #
        # processed_data = process_data(input_data)

        # Call model predict function
        #
        # predictions = model.predict(precessed_data)

        # Process the output
        #
        # output_data = postprocess_data(predictions)

        # Return data
        return output_data

    return model_api
