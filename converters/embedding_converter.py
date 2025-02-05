from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import base64
import pyperclip

# Load the face detection and recognition models
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to('cuda' if torch.cuda.is_available() else 'cpu')

saved_reference_embeddings = None

image = cv2.imread(input("Enter image file path: "))

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

embeddings, _ = find_face_encodings(image)
if embeddings is not None:

    # Convert embeddings to bytes and then to Base64
    embeddings_bytes = embeddings.tobytes()
    embeddings_base64 = base64.b64encode(embeddings_bytes).decode('utf-8')
    
    # Copy the Base64 string to clipboard
    pyperclip.copy(embeddings_base64)

    # Count rows
    rows = len(embeddings)
    # Count total elements
    total_elements = sum(len(row) for row in embeddings)
    print(f"Rows: {rows}, Total Elements: {total_elements}")
    
    # Print the size of the embeddings in bytes
    print(f"Size of embeddings in bytes: {embeddings.nbytes}")
else:
    print("No embeddings were created.")
