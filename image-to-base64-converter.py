import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')
    
image_path = "stored_faces/zach-selfie.jpg"
base64_string = image_to_base64(image_path)

print(base64_string)