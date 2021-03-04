
import cv2
import os


'''
This cell will turn on webcam and take photos, cropping a face if detected, 
and save it to directory ./datasets/name/name-x.png. 
'''

TEAMMATE_NAME = 'x'  # Insert name here!!
os.makedirs("./datasets/" + TEAMMATE_NAME)

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Load Haarcascade Frontal Face Classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Function returns cropped face
def face_extractor(photo):
    gray_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_photo, 1.1, 5)

    if faces is ():
        return None

    else:
        # Crop all faces found
        for (x, y, w, h) in faces:
            cropped_face = photo[y:y + h, x:x + w]

        return cropped_face


count = 0

# Collect 300 samples of your face from webcam input
while True:
    status, photo = cap.read()

    if face_extractor(photo) is not None:
        count += 1
        face = cv2.resize(face_extractor(photo), (224, 224))  # Save imgs as 224x224
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = './datasets/' + TEAMMATE_NAME + '/' + TEAMMATE_NAME + '-' + str(count) + '.png'  # saves images to ./datasets/name/name-x.png
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        pass

    if cv2.waitKey(1) == 13 or count == 300:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("Collecting Samples Complete")