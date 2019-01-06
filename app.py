from flask import Flask, render_template, request
from PIL import Image
import os
import model
import base64

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

# REST API send image file with tag 'image'
@app.route("/api", methods=['POST'])
def index():
    image_string = request.json['image']
    image = base64.b64decode(image_string)
    filename = "img"
    format_txt = ".jpg"
    path='uploaded_images/'+filename + format_txt
    imgFile = open(path, 'wb')
    imgFile.write(image)
    print('done')

    caption = model.generate_caption(os.path.join('uploaded_images', filename+format_txt))

    return caption

@app.route("/test")
def test():
    caption = model.generate_caption("images/i (5).jpg")
    
    return render_template('result.html', caption=caption, image="i (5).jpg")
    # return "Caption: "+caption

if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug=True)
    # app.run(debug=True)