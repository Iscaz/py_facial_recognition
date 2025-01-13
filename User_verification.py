from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import torch

app = Flask(__name__)

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MATCH_THRESHOLD = 70

# Models
mtcnn = MTCNN(keep_all=True, device=DEVICE)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

reference_embeddings = None  # Global variable for reference embeddings

# Function to find face encodings (embeddings)
def find_face_embeddings(image):
    try:
        boxes, _ = mtcnn.detect(image)
        if boxes is not None and len(boxes) > 0:
            print("Detected boxes:", boxes)  # Check detected face coordinates
            faces = [image[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in boxes]
            face_tensors = [
                torch.tensor(cv2.resize(face, (160, 160))).permute(2, 0, 1).float() / 255 for face in faces
            ]
            embeddings = model(torch.stack(face_tensors).to(DEVICE))
            return embeddings.detach().cpu().numpy(), boxes
        
        else:
            print("No faces detected in the image.")  
            return None, None
        
    except Exception as e:
        print(f"Error in find_face_encodings: {e}")
        return None, None


@app.route('/')
def index():
    return render_template('compare_embedding.html')  # Render the single page

# Route to compare embeddings against images.
# Requires embeddings and images sent as base64 strings in the POST request.
@app.route('/compare', methods=['POST'])
def compare():

    # Retrieve embedding and images from the form (allowing up to 5 embeddings and images)
    embedding_texts = [request.form.get(f'embedding {i}') for i in range(1, 6)]
    image_texts = [request.form.get(f'image {i}') for i in range(1, 6)]

    # Process the base64-encoded embedding
    embeddings = []

    for idx, embedding_text in enumerate(embedding_texts, start=1):
        if not embedding_text:
            continue

        try:
            embedding_bytes = base64.b64decode(embedding_text)
            reference_embeddings = np.frombuffer(embedding_bytes, np.float32).reshape(1, -1)
            embeddings.append({'id': f'embedding {idx}', 'value': reference_embeddings})

        except Exception as e:
            return jsonify({'similarity': 0, 'error': f"Error processing embedding: {str(e)}"})
    
    # Process base-64 image
    image_embeddings = []

    for idx, image_text in enumerate(image_texts, start=1):
        if not image_text:
            continue  # Skip if the image is not provided

        try:
            # Decode the image
            img_data = base64.b64decode(image_text)
            np_img = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            compared_embeddings, boxes = find_face_embeddings(image)

            image_embeddings.append({'id': f'image {idx}', 'value': compared_embeddings})

        except Exception as e:
            return jsonify({
                f'image {idx}': 0,
                'error': f"Error processing image {idx}: {str(e)}"
            })

    # Compare each embedding with each image, and add matches and similarities into a list
    matches = []
    similarities = []

    try:
        for embedding_data in embeddings:  # Each entry is {'id': 'embedding X', 'value': ...}
            embedding_id = embedding_data['id']
            reference_embeddings = embedding_data['value']

            if reference_embeddings is not None:
                for image_data in image_embeddings:  # Each entry is {'id': 'image X', 'value': ...}
                    image_id = image_data['id']
                    compared_embeddings = image_data['value']
                    
                    # Use absolute values to handle negative similarity scores from cosine similarity.
                    # Negative values can occur for non-similar images due to vector directionality.
                    similarity_score = abs(cosine_similarity(reference_embeddings, compared_embeddings)[0][0]) * 100

                    if similarity_score >= MATCH_THRESHOLD:
                        matches.append({
                            'embedding': embedding_id,
                            'image': image_id,
                            'similarity': round(similarity_score, 2),
                            'status': "match"
                        })
                    else:
                        similarities.append({
                            'embedding': embedding_id,
                            'image': image_id,
                            'similarity': round(similarity_score, 2)
                        })
            else:
                return jsonify({
                    f'image {idx}': 0,
                    'error': "No face detected or embeddings are None"
                })  
    except:
        return jsonify({'similarity': 0, 'error': 'Error with embedding'})

    if matches:
        return jsonify({'matches': matches})
    else:
        return jsonify({'status': "no match", 'similarities': similarities})

if __name__ == "__main__":
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # Suppress development server warning
    app.run(debug=True)

