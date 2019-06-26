from flask import Flask, render_template, request, jsonify
from PIL import Image
import os
import model_caption
import model_detection
import base64
import process_models_output
import uuid

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')


# REST API send image file with key 'image' and returns BOTH caption and tags
@app.route("/api/image", methods=['POST'])
def detect_tag_image():
    print('new tag & caption request received')
    image_string = request.json['image']
    print('decoding image file')
    image = base64.b64decode(image_string)

    file1 = str(uuid.uuid4())+".jpg"
    path='uploaded_images/' + file1
    imgFile = open(path, 'wb')
    imgFile.write(image)
    imgFile.close()

    file2 = str(uuid.uuid4())+".jpg"
    path2='uploaded_images/' + file2
    imgFile = open(path2, 'wb')
    imgFile.write(image)
    imgFile.close()

    print('saving image on the server is done')

    # get the tags
    try:
        objects = model_detection.start(os.path.join('uploaded_images', file1))
    except:
        objects = 'notags'
    objects = process_models_output.ProcessTags(objects)
    print('objects: '+objects)

    # get the caption
    try:
        caption = model_caption.generate_caption(os.path.join('uploaded_images', file2))
    except:
        caption = 'nocaption'
    caption = process_models_output.ProcessCaption(caption)
    print('caption: '+caption)

    # delete the image file
    os.remove(path)
    os.remove(path2)

    data = {
        'caption': caption,
        'tags': objects
    }

    return jsonify(data)


# REST API send image file with key 'image' and returns the tags
@app.route("/api/detection", methods=['POST'])
def detection():
    print('new tag request received')
    image_string = request.json['image']
    print('decoding image file')
    image = base64.b64decode(image_string)
    filename = str(uuid.uuid4())
    format_txt = ".jpg"
    path='uploaded_images/'+filename + format_txt
    imgFile = open(path, 'wb')
    imgFile.write(image)
    print('saving image temporarily is done')

    try:
        objects = model_detection.start(os.path.join('uploaded_images', filename+format_txt))
    except:
        objects = 'notags'
    objects = process_models_output.ProcessTags(objects)
    print('objects: '+objects)

    # delete the image file
    os.remove(path)

    data = {
        'tags': objects
    }

    return jsonify(data)


# REST API send image file with key 'image' and returns the caption
@app.route("/api/caption", methods=['POST'])
def caption():
    print('new image caption request received')
    image_string = request.json['image']
    image = base64.b64decode(image_string)
    filename = str(uuid.uuid4())
    format_txt = ".jpg"
    path='uploaded_images/'+filename + format_txt
    imgFile = open(path, 'wb')
    imgFile.write(image)
    print('saving image temporarily is done')

    try:
        caption = model_caption.generate_caption(os.path.join('uploaded_images', filename+format_txt))
    except:
        caption = 'nocaption'
    returncap = process_models_output.ProcessCaption(caption)
    print('caption: '+returncap)
    
    # delete the image file
    os.remove(path)

    data = {
        'caption':returncap
    }

    return jsonify(data)


@app.route("/testdetection")
def testdetection():
    objects = model_detection.start("images/testimg.jpg")
    objects = process_models_output.ProcessTags(objects)
    return render_template('result.html', caption=objects, image="testimg.jpg")


@app.route("/testcaption")
def testcaption():
    caption = model_caption.generate_caption("images/testimg.jpg")
    caption = process_models_output.ProcessCaption(caption)
    return render_template('result.html', caption=caption, image="testimg.jpg")


if __name__ == '__main__':
    # app.run(host= '0.0.0.0', debug=True)
    # app.run(debug=True)
    app.run(ssl_context=('cert.pem', 'key.pem'), host='0.0.0.0', debug=True)