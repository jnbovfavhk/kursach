import HaarBasedMethods
import HogMethods
import joblib
import os
import DataReduction


def demonstrate_haar():
    images_path, labels_path = DataReduction.reduct_images_dataset("wider_face/images",
                                                                   "wider_face/labels", 10)

    if not os.path.isfile('haar_trained.pkl'):
        model = HaarBasedMethods.get_trained_model(images_path, labels_path)
        joblib.dump(model, 'haar_trained.pkl')
        print("Модель, основанная на каскадах Хаара, сериализована")
    else:
        model = joblib.load('haar_trained.pkl')
        print("Модель десериализована")

    HaarBasedMethods.detect_and_draw("wider_face/images/wider_2236.jpg",
                               model, "ProcessedImages/testImageHaar.jpg")
    print("Обнаружены лица на тестовом изображении")

def demonstrate_hog():
    images_path, labels_path = DataReduction.reduct_images_dataset(
        "wider_face/images",
        "wider_face/labels", 10)

    if not os.path.isfile('hog_trained.pkl'):
        model = HogMethods.get_trained_model(images_path, labels_path)
        joblib.dump(model, 'hog_trained.pkl')
        print("Модель HOG сериализована")
    else:
        model = joblib.load('hog_trained.pkl')
        print("Модель Hog десериализована")

    HogMethods.detect_and_draw("wider_face/images/wider_2236.jpg",
                               model, "ProcessedImages/testImageHog.jpg")
    print("Обнаружены лица на тестовом изображении")



if __name__ == '__main__':
    # demonstrate_hog()
    demonstrate_haar()