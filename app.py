from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)

@app.route('/sketch', methods=['POST'])
def sketch():
    if 'image' not in request.files:
        return {"error": "No image uploaded"}, 400
    
    file = request.files['image']
    
    # Read the image from the request
    image_stream = file.read()
    np_image = np.frombuffer(image_stream, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive histogram equalization to enhance details
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    grey_img = clahe.apply(grey_img)
    
    # Edge-preserving filter to enhance details
    image_smooth = cv2.bilateralFilter(grey_img, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Inverting and blurring the image
    invert = cv2.bitwise_not(image_smooth)
    blur = cv2.GaussianBlur(invert, (15, 15), 0)
    inverted_blur = cv2.bitwise_not(blur)
    
    # Create the sketch
    sketch = cv2.divide(grey_img, inverted_blur, scale=256.0)
    
    # Convert sketch image to byte stream
    _, buffer = cv2.imencode('.jpg', sketch)
    img_io = io.BytesIO(buffer)
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
