import base64
import pyperclip

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')
    
image_path = input("Enter file path: ")
base64_string = image_to_base64(image_path)

pyperclip.copy(base64_string)