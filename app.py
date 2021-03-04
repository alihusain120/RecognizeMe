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


@app.route('/')
def home():
    return render_template('./index.html')


@app.route('/test',methods=['GET'])
def test():
    return "hello world!"


@app.route('/submit',methods=['POST'])
def submit():
    image = request.form['video_feed']
    base64_img = image[22:]

    # decoded = decoded_bytes.decode('ascii')
    print('start of base64, substr: ' + str(image.find('base64,')))
    print('char at 15: ' + str(image[15]))
    print(image)
    print(base64_img)

    decoded_bytes = base64.b64decode(image)

    with open("savedImage.png", "wb") as fh:
        fh.write(decoded_bytes)

    # TODO: process the image as you see fit here to ensure the system recognizes
    # you and your teammates. Bonus points if you can prevent the system from being fooled by someone
    # holding up a photo of you or your teammates to the webcam, though this is not required.

    # For now, render the logged in page if the user is logged in.

    # TODO: configure this call to FaceIdentify to accept or reject a identified face
    face = FaceIdentify(precompute_features_file="./datasets/precompute_features.pickle")
    face.detect_face()

    if image:
        return render_template("logged_in.html")
    return render_template("unauthorized.html")


if __name__ == "__main__":
    app.run(debug=True)
