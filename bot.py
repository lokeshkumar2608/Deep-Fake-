import io
from flask import Flask, request, jsonify
from transformers import pipeline

def create_app():
    app = Flask(__name__)

    # Initialize the audio classification pipeline
    pipe = pipeline("audio-classification", model="mo-thecreator/Deepfake-audio-detection")

    @app.route('/predict', methods=['POST'])
    def predict():
        # Check if the request contains a file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        # Ensure the file has a name
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        try:
            # Read the audio file as bytes
            audio_bytes = file.read()

            # Use the pipeline to classify the audio
            outputs = pipe(audio_bytes)

            # Extract outputs into variables
            labels = [output['label'] for output in outputs]
            scores = [output['score'] for output in outputs]

            # Get top-1 prediction
            top_label = labels[0]
            top_score = scores[0]

            # Return the prediction result as a JSON response
            return jsonify({
                'predicted_label': top_label,
                'confidence_score': top_score
            })

        except Exception as e:
            # Return any errors that occur during the process
            return jsonify({'error': str(e)}), 500

    return app

# Run the Flask app if this script is executed
if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5010, debug=True)
