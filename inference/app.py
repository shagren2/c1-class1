import os
import uuid
import json
from flask_cors import CORS

from flask import (Flask, request)
from decouple import config
from platform_classification.inference.inference_model import InferenceModel
from platform_classification.common import get_checkpoint_name

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config('UPLOAD_FOLDER')
CORS(app)
PORT = config('PORT', cast=int)
INFERENCE_PATH_TO_CHECKPOINT = config('INFERENCE_PATH_TO_CHECKPOINT')
TRAIN_PATH_TO_SAVE_RESULT = config('TRAIN_PATH_TO_SAVE_RESULT')


inference_path = get_checkpoint_name(INFERENCE_PATH_TO_CHECKPOINT, TRAIN_PATH_TO_SAVE_RESULT)
inference_model = InferenceModel(path_to_checkpoint=inference_path)


def save_image(request):
    file = request.files['file']
    extension = os.path.splitext(file.filename)[1]
    f_name = str(uuid.uuid4()) + extension
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], f_name)
    file.save(full_path)
    return full_path


@app.route('/api/health_check', methods=['GET'])
def health_check():
    return "OK"


@app.route('/api/classify', methods=['POST'])
def image_net_classify():
    if request.method == 'POST':
        full_path = save_image(request)
        result = inference_model.classify(full_path)
        return json.dumps(result)

if __name__ == '__main__':
    # Only for debug purpose
    app.run(host='0.0.0.0', port=PORT, debug=False)
