# Image Captioning Deep Learning Model Deployed on Flask Server

### Setup

1. Clone repo

2. Move files 'keras_model.h5' to project folder and 'captions_train2014.json' 'captions_train2014.json' to data/coco/annotations

3. Move "yolo.h5" file to data folder

4. Create virtual enviornment named "venv": `virtualenv venv`

5. Activate virtual enviornment
    for windows: `venv\Scripts\activate.bat`
    for linux/macOS: `source venv/bin/activate`

6. Install project dependencies: `pip install -r requirements.txt`

7. Run flask server: `python app.py`
