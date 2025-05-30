import torch

import CNNMethods
import HaarBasedMethods
import HogMethods
import joblib
import os
import DataReduction
import SiftMethods


def demonstrate_haar():

    images_path, labels_path = DataReduction.reduct_images_dataset("wider_face/images",
                                                                   "wider_face/labels", 1000)

    if not os.path.isfile('haar_trained.pkl'):
        model = HaarBasedMethods.get_trained_model(images_path, labels_path)
        joblib.dump(model, 'haar_trained.pkl')
        print("Модель, основанная на каскадах Хаара, сериализована")
    else:
        model = joblib.load('haar_trained.pkl')
        print("Модель десериализована")

    HaarBasedMethods.detect_and_draw("wider_face/images/wider_2236.jpg",
                               model, "ProcessedImages/testImageHaarAt1000TrainImages.jpg")
    print("Обнаружены лица на тестовом изображении")

def demonstrate_hog():
    images_path, labels_path = DataReduction.reduct_images_dataset(
        "wider_face/images",
        "wider_face/labels", 1000)

    if not os.path.isfile('hog_trained.pkl'):
        model = HogMethods.get_trained_model(images_path, labels_path)
        joblib.dump(model, 'hog_trained.pkl')
        print("Модель HOG сериализована")
    else:
        model = joblib.load('hog_trained.pkl')
        print("Модель Hog десериализована")

    HogMethods.detect_and_draw("wider_face/images/wider_2236.jpg",
                               model, "ProcessedImages/testImageHogAt1000TrainImages.jpg")
    print("Обнаружены лица на тестовом изображении")

def demonstrate_sift():
    images_path, labels_path = DataReduction.reduct_images_dataset(
        "wider_face/images",
        "wider_face/labels", 1000)


    if not os.path.isfile('sift_trained.pkl'):
        model = SiftMethods.get_trained_model(images_path, labels_path)
        model.save_model("sift_trained.pkl")
        print("Модель SIFT сериализована")
    else:
        model = SiftMethods.FaceDetector()
        model = model.load_model("sift_trained.pkl")
        print("Модель Sift десериализована")

    model.detect_and_draw("wider_face/images/wider_2236.jpg", "ProcessedImages/testImageSIFTAt1000TrainImages.jpg")
    print("Обнаружены лица на тестовом изображении")

def demonstrate_cnn():
    images_path, labels_path = DataReduction.reduct_images_dataset(
        "wider_face/images",
        "wider_face/labels", 1000)
    if not os.path.isfile('cnn_trained.pt'):
        model = CNNMethods.get_trained_model(images_path, labels_path)
        CNNMethods.save_model(model, 'cnn_trained.pt')
    else:
        model = CNNMethods.load_model('cnn_trained.pt')
    CNNMethods.detect_and_draw(model, "wider_face/images/wider_2236.jpg", "ProcessedImages/testImageCNNAt100TrainImages.jpg")


if __name__ == '__main__':
    # demonstrate_hog()

    # demonstrate_haar()
    demonstrate_sift()
    # demonstrate_cnn()
