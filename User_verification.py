from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import torch

app = Flask(__name__)

# Load the face detection and recognition models
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to('cuda' if torch.cuda.is_available() else 'cpu')

saved_reference_embeddings = None  # Global variable for reference embeddings

# Function to find face encodings (embeddings)
def find_face_encodings(image):
    try:
        boxes, _ = mtcnn.detect(image)
        if boxes is not None and len(boxes) > 0:
            print("Detected boxes:", boxes)  # Check detected face coordinates
            faces = [image[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in boxes]
            face_tensors = [
                torch.tensor(cv2.resize(face, (160, 160))).permute(2, 0, 1).float() / 255 for face in faces
            ]
            embeddings = model(torch.stack(face_tensors).to('cuda' if torch.cuda.is_available() else 'cpu'))
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

@app.route('/compare', methods=['POST'])
def compare():
    global saved_reference_embeddings

    # Retrieve embedding and images from the form
    embedding_text = request.form.get('embedding')
    image_texts = [request.form.get(f'image{i}') for i in range(1, 5)]  # Add more range values for more images

    # Process the base64-encoded embedding
    try:
        embedding_bytes = base64.b64decode(embedding_text)
        saved_reference_embeddings = np.frombuffer(embedding_bytes, np.float32).reshape(1, -1)
    except Exception as e:
        return jsonify({'similarity': 0, 'error': f"Error processing embedding: {str(e)}"})

    # Process and compare each image
    similarities = []
    for idx, image_text in enumerate(image_texts, start=1):
        if not image_text:
            continue  # Skip if the image is not provided

        try:
            # Decode the image
            img_data = base64.b64decode(image_text)
            np_img = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            # Find face encodings
            compared_embeddings, boxes = find_face_encodings(image)
            if compared_embeddings is not None:
                # Calculate similarity score
                similarity_score = cosine_similarity(saved_reference_embeddings, compared_embeddings)[0][0] * 100

                # Check if there's a match
                if similarity_score >= 70:
                    return jsonify({
                        'status': "match",
                        f'image {idx}': round(similarity_score, 2)
                    })

                similarities.append({
                    f'image {idx}': round(similarity_score, 2)
                })
            else:
                similarities.append({
                    f'image {idx}': 0,
                    'error': "No face detected or embeddings are None"
                })
        except Exception as e:
            similarities.append({
                f'image {idx}': 0,
                'error': f"Error processing image {idx}: {str(e)}"
            })

    # If no matches are found, return all similarity scores
    if similarities:
        return jsonify({
            'status': "no match",
            'similarities': similarities
        })

    return jsonify({'similarity': 0, 'error': 'No valid images provided'})


if __name__ == "__main__":
    app.run(debug=True)
