import base64
import torch
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, redirect
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import io

# Initialize Flask app
app = Flask(__name__)

# Initialize models
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=DEVICE)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# Store the pre-given embedding
stored_embedding = None

def find_face_embeddings(image):
    try:
        boxes, _ = mtcnn.detect(image)
        if boxes is not None and len(boxes) > 0:
            faces = [image[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in boxes]
            face_tensors = [
                torch.tensor(cv2.resize(face, (160, 160))).permute(2, 0, 1).float() / 255 for face in faces
            ]
            embeddings = model(torch.stack(face_tensors).to(DEVICE))
            return embeddings.detach().cpu().numpy(), boxes.tolist()
        else:
            return None, None
    except Exception as e:
        print(f"Error in find_face_embeddings: {e}")
        return None, None

@app.route('/')
def home():
    return redirect('/upload_embedding')

@app.route('/upload_embedding', methods=['GET'])
def upload_embedding_page():
    return render_template('upload_embedding.html')

@app.route('/upload_embedding', methods=['POST'])
def upload_embedding():
    global stored_embedding
    try:
        base64_embedding = request.form['embedding']
        binary_data = base64.b64decode(base64_embedding)
        stored_embedding = np.frombuffer(binary_data, dtype=np.float32)
        return redirect('/index')
    except Exception as e:
        return jsonify({"error": f"Failed to upload embedding: {e}"}), 400

@app.route('/get_similarity', methods=['POST'])
def get_similarity():
    try:
        frame = request.files['frame'].read()
        np_arr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        embeddings, boxes = find_face_embeddings(img)
        
        if embeddings is not None and stored_embedding is not None:
            similarity = (1 - cosine(embeddings[0], stored_embedding)) * 100
            return jsonify({"similarity": similarity, "boxes": boxes}), 200
        else:
            return jsonify({"similarity": None, "boxes": []}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to calculate similarity: {e}"}), 400

@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
