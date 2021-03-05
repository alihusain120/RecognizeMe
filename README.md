# RecognizeMe
Simple web app to detect and recognize faces

Main work is done in model/

__RecognizeMe.ipynb__:
  This has all the main functionality written and tested that was then translated onto the .py files. For some reason keras_vggface module doesn't load in the .py scripts sometimes, so if replicating it try to use this notebook's cells rather than corresponding .py files.

__sample_collect.py__:

  Runs a script with openCV to snap 300 photos from system webcam. Each photo is detected for a face using OpenCV's Cascade Classifier. If a face is detected it is cropped and saved. 
  
__precompute_features__:

  Runs through each image for each team member inside datasets/ (gitignored the image datasets). For each person, this file computes the feature map using the resnet50 model from VGGFace2.
  The top layer of the resnet50 model is not set, so our output is a tensor of features. This tensor is the mean feature map for each team member, and it is saved as a numpy array.
  
  All the {Name; Features} pairs are saved and pickled
  
  
__face_identify.py__:

  Main class to run the face detection and recognition. Tries to detect face through webcam, and if successful, compares the face to each of the saved precomputed feature maps and returns result.
