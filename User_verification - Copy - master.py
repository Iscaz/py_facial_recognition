from flask import Flask, render_template, request, jsonify, redirect, url_for
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

# Creating the function to find face encodings (embeddings)
def find_face_encodings(image):
    # Get face bounding boxes and align the faces
    boxes, _ = mtcnn.detect(image)

    if boxes is not None:
        # Crop and resize faces and convert into embeddings
        faces = [image[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in boxes]
        face_tensors = [torch.tensor(cv2.resize(face, (160, 160))).permute(2, 0, 1).float() / 255 for face in faces]
        embeddings = model(torch.stack(face_tensors).to('cuda' if torch.cuda.is_available() else 'cpu'))
        return embeddings.detach().cpu().numpy(), boxes  # Return embeddings and bounding boxes
        
    else:
        return None, None

# Real-time face verification route
# Detecting scanned face and finding its embeddings
# Comparing the newly found embeddings with the embeddings obtained earlier
@app.route('/verify_live', methods=['POST'])
def verify_live():
    global saved_reference_embeddings
    data = request.json

    # Decode the base64 frame from the webcam
    frame_data = data['frame'].split(',')[1]
    img_data = base64.b64decode(frame_data) # Decodes the base64 encoded string into raw binary (bytes)
    np_img = np.frombuffer(img_data, np.uint8) # Convert byte data (img_data) to NumPy array in unsigned 8-bit integers
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)  # Convert to an image
    
    # Get the encodings of the captured image
    compared_embeddings, boxes = find_face_encodings(image)

    if compared_embeddings is not None and saved_reference_embeddings is not None:
        results = []  # Initialize results as an empty list
        similarities = []  # List to store similarity scores

        for i, box in enumerate(boxes):
            # Calculate cosine similarity for each detected face's embedding
            similarity_score = cosine_similarity(saved_reference_embeddings, compared_embeddings[i])
            similarity = similarity_score[0][0] * 100

            # Convert bounding box coordinates to integers for pixel accuracy
            box = [int(b) for b in box]

            # Append the similarity and bounding box for this face to results
            results.append({
                'similarity': round(similarity, 2),
                'box': box  # Box coordinates as [x, y, width, height]
            })

            # Append similarity score to similarities list for reshaping later
            similarities.append(round(similarity, 2))

        print("Similarities before reshaping:", similarities)  # Debugging print

        if not similarities:
            return jsonify({'error': 'No similarities calculated, no faces detected.'})

        # Now, reshape the similarities array to be 2D
        reshaped_similarities = np.array(similarities).reshape(-1, 1)

        # Return JSON response with the reshaped similarity scores
        return jsonify({
            'similarities': reshaped_similarities.tolist(),  # Convert to list for JSON serialization
            'results': results
        })

    else:
        # Return if no face found or no reference embeddings
        return jsonify([{'similarity': 0, 'box': None}])



# Paste and submit embedding (first page)
@app.route('/')
def paste_embedding():
    return render_template('paste_embedding.html')

@app.route('/submit_embedding', methods=['POST'])
def submit_embedding():
    embedding_text = request.form['embedding']
    embedding_text2 = request.form['embedding2']
    
    try:
        # Decode pasted base64 string into raw bytes, then into NumPy array of float 32 value for both embeddings
        embedding_bytes = base64.b64decode(embedding_text)
        reference_embeddings = np.frombuffer(embedding_bytes, np.float32).reshape(1, -1) # Reshape as necessary for cosine similarity
        
        embedding_bytes2 = base64.b64decode(embedding_text2)
        reference_embeddings2 = np.frombuffer(embedding_bytes2, np.float32).reshape(1, -1)
        
        global saved_reference_embeddings
        saved_reference_embeddings = np.vstack([reference_embeddings, reference_embeddings])  # Store embeddings for later comparison

        return redirect(url_for('selfie'))
        # A URL is dynamically created for the 'selfie' route and the user is redirected to the 'selfie' page once string is successfully converted

    except Exception as e:
        return f"Error processing embeddings: {str(e)}"

@app.route('/selfie')
def selfie():
    return render_template('camera.html')

if __name__ == "__main__":
    app.run(debug=True)
