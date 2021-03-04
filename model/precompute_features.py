
# This code adapted from this blog tutorial:
# https://www.dlology.com/blog/live-face-identification-with-pre-trained-vggface2-model/
# GithubRepo of original author: https://github.com/Tony607/Keras_face_identification_realtime


from keras.preprocessing import image
import numpy as np
from keras_vggface import utils

import os
import glob
import pickle


DATASETS = "./datasets/"


# Helper method to save byte stream of name/features
def pickle_stuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()


def main():

    # Load VGGFace for computing face feature vectors
    resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                                pooling='avg')  # pooling: None, avg or max

    def image2x(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)  # or version=2
        return x

    # Method to calculate mean features map on folder of person images (labeled name)
    def cal_mean_feature(image_folder):
        face_images = list(glob.iglob(os.path.join(image_folder, '*.*')))
        print(str(len(face_images)))

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        batch_size = 32
        face_images_chunks = chunks(face_images, batch_size)
        fvecs = None
        for face_images_chunk in face_images_chunks:
            images = np.concatenate([image2x(face_image) for face_image in face_images_chunk])
            print("image len: " + str(len(images)))
            batch_fvecs = resnet50_features.predict(images)  # Use VGGFace model to get feature map on images

            if fvecs is None:
                fvecs = batch_fvecs
            else:
                fvecs = np.append(fvecs, batch_fvecs, axis=0)  # Append all feature vectors
        return np.array(fvecs).sum(axis=0) / len(fvecs)  # Return mean feature map as np array

    folders = list(glob.iglob(os.path.join(DATASETS, '*')))
    names = [os.path.basename(folder) for folder in folders]

    '''
    for i, folder in enumerate(folders):
        name = names[i]
        videos = list(glob.iglob(os.path.join(folder, '*.*')))
        save_folder = os.path.join(FACE_IMAGES_FOLDER, name)
        print(save_folder)
        os.makedirs(save_folder, exist_ok=True)
        for video in videos:
            extractor.extract_faces(video, save_folder)
    '''

    # for each folder of a teammate
    # send image folder to cal_mean_feature() to calc feature map
    # save {"name": teammate_name, "features": mean_feature_map}
    # pickle it ()
    precompute_features = []
    for i, folder in enumerate(folders):
        name = names[i]
        save_folder = os.path.join(DATASETS, name)
        print("save folder: " + str(save_folder))
        mean_features = cal_mean_feature(image_folder=save_folder)
        precompute_features.append({"name": name, "features": mean_features})

    # https://stackoverflow.com/questions/8968884/python-serialization-why-pickle
    pickle_stuff("./datasets/precompute_features.pickle", precompute_features)

if __name__ == "__main__":
    main()