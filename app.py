# Main Application Code

from flask import Flask, render_template, request, flash, redirect, url_for
import pickle
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import os
from PIL import Image, ImageOps
from keras.preprocessing import image
from keras.models import Model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

app = Flask(__name__)

UPLOAD_FOLDER = 'static/images/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

model_rfc = pickle.load(open('model_rfc_pickle', 'rb'))
# model_svm = pickle.load(open('model_svm_pickle', 'rb'))

dic = {0: 'Normal', 1: 'Infected'}


@app.route('/')
def home():
    return render_template('Index.html')


@app.route('/Index.html')
def index():
    return render_template('Index.html')


@app.route('/Test_Yourself.html')
def contact():
    return render_template('Test_Yourself.html')


@app.route('/Prevention.html')
def news():
    return render_template('Prevention.html')


@app.route('/About_Us.html')
def about():
    return render_template('About_Us.html')


def predict_label(img_path):
    size = (192, 192)
    img1 = image.load_img(img_path, target_size=(192, 192))

    image1 = ImageOps.fit(img1, size, Image.ANTIALIAS)
    image2 = np.asarray(image1)
    img = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    xn, yn, rgb = img.shape
    image3 = img.reshape((1, xn * yn * rgb))
    final_1 = model_rfc.predict(image3)
    # final_2 = model_svm.predict(image3)
    final = (final_1[0])

    if final > 0.5:
        final = 1
    else:
        final = 0

    return dic[final]


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)

    return render_template("Test_Yourself.html", prediction=p, img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)
