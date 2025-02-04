# Face Similarity Web App

This is a Flask-based web application that performs facial recognition by comparing face embeddings using the FaceNet model. It detects faces in images and computes similarity scores based on cosine similarity.

## Features

- Detects and extracts faces using MTCNN.
- Generates face embeddings using FaceNet (InceptionResnetV1 pretrained on VGGFace2).
- Compares facial embeddings using cosine similarity.
- Returns similarity scores and identifies matches based on a set threshold.

## Requirements

Ensure you have Python installed, then install the dependencies:

```bash
pip install flask numpy opencv-python facenet-pytorch torch scikit-learn
```

## Program Structure

This project consists of two separate APIs:

1. **Main_User_Verification.py**: Compares multiple embeddings with multiple photos to determine similarity scores.
2. **app.py**: Performs real-time facial recognition using a webcam feed with the help of HTML files.

## Usage

### Running Main_User_Verification.py

1. **Start the API**
   ```bash
   python Main_User_Verification.py
   ```
2. **Send a request to compare embeddings with images** (see API Endpoints below).

### Running Live Face Recognition (`app.py`)

1. **Insert a base64-encoded face embedding** as a reference before running `app.py`.
2. **Start the live facial recognition application**
   ```bash
   python app.py
   ```
3. **Access the webcam interface** Open `http://127.0.0.1:5000/` in your browser.
4. **Live face scanning:** The application will highlight detected faces and display similarity scores in real-time.

## Generating Base64 Data

Use the `embedding_converter.py` and `photo_converter.py` files to generate the base64 data needed for requests. Simply enter the file path of the desired photo for conversion.

## API Endpoints

### `GET /`

Serves the main comparison page.

### `POST /compare`

Compares uploaded face embeddings against images.

#### Request Parameters (Form Data):

- `embedding 1`, `embedding 2`, ..., `embedding 5`: Base64-encoded face embeddings.
- `image 1`, `image 2`, ..., `image 5`: Base64-encoded images.

#### Response Format:

- **Match found:**
  ```json
  {
    "matches": [
      {"embedding": "embedding 1", "image": "image 1", "similarity": 85.32, "status": "match"}
    ]
  }
  ```
- **No match:**
  ```json
  {
    "status": "no match",
    "similarities": [
      {"embedding": "embedding 1", "image": "image 2", "similarity": 65.47}
    ]
  }
  ```

## Configuration

- **Threshold:** The match threshold is set to `70%` similarity.
- **Device:** Uses GPU (`cuda`) if available, otherwise defaults to CPU.

## Notes

- Ensure input images are clear and well-lit for accurate results.
- The application supports up to 5 reference embeddings and 5 images per request.

## License

This project is open-source. Modify and use it as needed.

