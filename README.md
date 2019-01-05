# Image Captioning Deep Learning Model Deployed on Flask Server

### Setup

1. Clone repo

3. Move files 'keras_model.h5' to project folder and 'captions_train2014.json' 'captions_train2014.json' to data/coco/annotations

4. Create virtual enviornment named "venv" or anything else: `virtualenv venv`

5. Activate virtual enviornment
    for windows: `virtualenv\Scripts\activate.bat`
    for linux/macOS: `source virtualenv/bin/activate`

6. Install project dependencies: `pip install -r requirements.txt`

7. Run flask server: `python app.py`
