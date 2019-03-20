from flask import Flask, render_template, request
from PIL import Image
import os
import model_caption
import model_detection
import base64

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/api/detection", methods=['POST'])
def detection():
    print('new object detection request received')
    image_string = request.json['image']
    image = base64.b64decode(image_string)
    filename = "img"
    format_txt = ".jpg"
    path='uploaded_images/'+filename + format_txt
    imgFile = open(path, 'wb')
    imgFile.write(image)
    print('saving image temporarily is done')

    objects = model_detection.start(os.path.join('uploaded_images', filename+format_txt))
    print('objects: '+objects)
    
    return objects

# REST API send image file with tag 'image'
@app.route("/api/caption", methods=['POST'])
def caption():
    print('new image caption request received')
    image_string = request.json['image']
    image = base64.b64decode(image_string)
    filename = "img"
    format_txt = ".jpg"
    path='uploaded_images/'+filename + format_txt
    imgFile = open(path, 'wb')
    imgFile.write(image)
    print('saving image temporarily is done')

    caption = model_caption.generate_caption(os.path.join('uploaded_images', filename+format_txt))
    captionWords = caption.split()
    finalCaption = []
    if(captionWords[-1]=="eeee"):
        for i in range(0,len(captionWords)-1):
            finalCaption.append(captionWords[i])

    returncap = " ".join(finalCaption)
    print('caption: '+returncap)
    
    return returncap

@app.route("/testdetection")
def testdetection():
    objects = model_detection.start("images/img.jpg")
    
    return render_template('result.html', caption=objects, image="img.jpg")


@app.route("/testcaption")
def testcaption():
    caption = model_caption.generate_caption("images/img.jpg")
    
    return render_template('result.html', caption=caption, image="img.jpg")


if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug=True)
    # app.run(debug=True)