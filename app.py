from flask import Flask, request, jsonify
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import io

def create_app():
    app = Flask(__name__)

    # Load model and processor
    model = ViTForImageClassification.from_pretrained("ViT_Deepfake_Detection")
    processor = ViTImageProcessor.from_pretrained("ViT_Deepfake_Detection")
    model.eval()

    @app.route('/predict', methods=['POST'])
    def predict():
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']

        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = torch.argmax(logits, dim=1).item()

            label = model.config.id2label[predicted_class_idx]

            return jsonify({'predicted_label': label})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app

# Server startup
if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5009, debug=True)
