# Code provided here was obtained from: https://stackoverflow.com/a/54808032/4822073
# Original Author: Jacob Lawrence <https://stackoverflow.com/users/8736261/jacob-lawrence>
# Licensed under CC-BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
# Minor modifications made by Dharmesh Tarapore <dharmesh@cs.bu.eu>
from flask import Flask,request,jsonify, render_template
import numpy as np
import cv2
import base64

from model.face_identify import FaceIdentify

app = Flask(__name__, template_folder="views")

CASE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_size = 224


@app.route('/')
def home():
    return render_template('./index.html')


@app.route('/test',methods=['GET'])
def test():
    return "hello world!"


@app.route('/submit',methods=['POST'])
def submit():
    face = FaceIdentify(precompute_features_file="./model/precompute_features.pickle")
    image = request.form['video_feed']
    name = face.attempt_single_login()
    print(name)

    if name != "UNRECOGNIZED":
        return render_template("logged_in.html",name = name)
    return render_template("unauthorized.html")


if __name__ == "__main__":
    app.run(debug=True)
