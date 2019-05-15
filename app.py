import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

from serve import get_model_api

# Define the app
app = Flask(__name__)
CORS(app)  # needed for cross-domain requests, allow everything by default

# Logging for Heroku
if "DYNO" in os.environ:
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.INFO)


# Load the model
model_api = get_model_api()

# API route
@app.route("/api", methods=["POST"])
def api():
    """API function.

    Returns:
        type: Description of returned object.
    """
    input_data = request.json

    app.logger.info(f"api_input: {str(input_data)}")

    output_data = model_api(input_data)

    app.logger.info(f"api_output: {str(output_data)}")

    response = jsonify(output_data)

    return response


@app.route("/")
def index():
    return "Index API"


# HTTP Error handlers
@app.errorhandler(404)
def url_error(e):
    return f"""Wrong URL! <pre>{e}</pre>""", 404


@app.errorhandler(500)
def server_error(e):
    return (
        f"""An internal error occured: <pre>{e}</pre>. See logs for full stacktrace""",
        500,
    )


if __name__ == "__main__":
    # This is used when running locally
    app.run(host="0.0.0.0", debug=True)
