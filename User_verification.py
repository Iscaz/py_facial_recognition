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
    # Allows variable to be used outside this fucntion
    data = request.json
    # A python dictionary is created called data to store JSON data sent in the POST request for further conversion later

    # Decode the base64 frame from the webcam
    frame_data = data['frame'].split(',')[1]
    # data is an object that is accessing the 'frame' key to retrieve the respective values
    # split splits the substring to access the second part of the string (by using index 1) which would be after the comma delimiter
    
    img_data = base64.b64decode(frame_data)
    # this function decodes the base64 encoded string into raw binary (bytes)

    np_img = np.frombuffer(img_data, np.uint8)
    # NumPy array is created from the converted byte data
    # np.frombuffer contains two arguments, first is input(raw data) and second is the output and its specified datatype
    # np.uint8 indicates that the data type is unsigned 8-bit integers (standard representation for image pixel values (0-255))
    # This step makes np_img into a one-dimensional NumPy array with pixel values of the image

    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)  # Convert to an image
    # np_img is then converted into image format so OpenCV can work with it through imdecode function
    # Second parameter cv2.IMREAD_COLOR tells OpenCV to load the image in color (converts 1D array of pixel into 3D array for color images for further processing)

    # Get the encodings of the captured image
    compared_embeddings, boxes = find_face_encodings(image)
    # Using the function created previously, embeddings of the detected face and the bounding box of that face is returned

    if compared_embeddings is not None and saved_reference_embeddings is not None:
        # Calculate cosine similarity
        similarity_score = cosine_similarity(saved_reference_embeddings, compared_embeddings)
        # cosine_similarity function compares the similarity between the two vectors passed through

        similarity = similarity_score[0][0] * 100 
        # Since the result of cosine_similarity is a 2D array, the similarity_score[0][0] extracts the actual similarity value
        # Times by 100 to convert to percentage

        # Prepare the bounding box (assuming we're only tracking the first detected face)
        box = boxes[0] if len(boxes) > 0 else None
        # Selecting the first face detected (index 0), as long as there is more than one face

        box = [int(b) for b in box] if box is not None else None
        # bounding box coordinates are converted into integer for pixel accuracy

        # Return the similarity percentage and the bounding box
        return jsonify({
            'similarity': round(similarity, 2),
            # Return the similarity percentage that is rounded to 2 decimal places

            'box': box  # Send the box as [x, y, width, height] to draw rectangle around the detected face
        })
    else:
        # Return no face found or no reference embeddings
        return jsonify({'similarity': 0, 'box': None})

# Route to paste embedding (first page)
@app.route('/')
# Calls paste_embeddings function that returns the 'paste_embedding' html page
def paste_embedding():
    return render_template('paste_embedding.html')

@app.route('/submit_embedding', methods=['POST'])
# Calls submit_embeddings function that creates an object (embedding_text) that gets its value from the embedding form from the http post request
def submit_embedding():
    embedding_text = request.form['embedding']
    
    try:
        # Decode base64 text into NumPy array
        embedding_bytes = base64.b64decode(embedding_text)
        # Decodes the embedding_text from base64 string back into raw bytes
        reference_embeddings = np.frombuffer(embedding_bytes, np.float32)
        # Converts the bytes into NumPy array of float32 value (second argument as the output datatype)
        reference_embeddings = reference_embeddings.reshape(1, -1)  
        # Reshape as necessary for cosine similarity
        
        global saved_reference_embeddings
        saved_reference_embeddings = reference_embeddings  # Store embeddings for later comparison

        return redirect(url_for('selfie'))
        # A URL is dynamically created for the 'selfie' route and the user is redirected to the 'selfie' page 

    except Exception as e:
        return f"Error processing embeddings: {str(e)}"

@app.route('/selfie')
def selfie():
    return render_template('camera.html')

if __name__ == "__main__":
    app.run(debug=True)
