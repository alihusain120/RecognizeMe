# RecognizeMe
Simple web app to detect and recognize faces

Main work is done in model/

__RecognizeMe.ipynb__:
  This has all the main functionality written and tested that was then translated onto the .py files. For some reason keras_vggface module doesn't load in the .py scripts sometimes, so if replicating it try to use this notebook's cells rather than corresponding .py files.

__sample_collect.py__:

  Runs a script with openCV to snap 300 photos from system webcam. Each photo is detected for a face using OpenCV's Cascade Classifier. If a face is detected it is cropped and saved. 
  
__precompute_features__:

  Runs through each image for each team member inside datasets/ (gitignored the image datasets). For each person, this file computes the feature map for each photo of that person using the resnet50 model from VGGFace2.
  The top layer of the resnet50 model is not set, so our output is a tensor of features. The mean of the tensors for each photo taken, and the single mean features tensor for each team member is saved as a numpy array.
  
  All the {Name; FeaturesnpArray} pairs are saved and pickled
  
  
__face_identify.py__:

  Main class with functionality to run the face detection and recognition. Tries to detect face through webcam, and if successful, compares the face to each of the saved precomputed feature maps and returns result.
  
  __app.py__

  app.py is up-to-date on the __development branch__ only, not the master/main. One can run app.py and test on localhost.


ATTRIBUTIONS: 

This project and some code (attributed in the .py files respectively) was adapted from the following sources: 

1. https://www.dlology.com/blog/live-face-identification-with-pre-trained-vggface2-model/ and the corresponding GitHub repo: https://github.com/Tony607/Keras_face_identification_realtime.

2. https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/

3. https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/

4. https://medium.com/@chiraggoelit/face-recognition-using-transfer-learning-9986728c443d
