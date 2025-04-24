import HaarBasedMethods
import HogMethods
import joblib
import os
import DataReduction


def demonstrate_haar():
    images_path, labels_path = DataReduction.reduct_images_dataset("C:\\Users\\ilyab\\Downloads\\wider_face\\images",
                                                                   "C:\\Users\\ilyab\\Downloads\\wider_face\\labels", 100)

    if not os.path.isfile('haar_trained.pkl'):
        model = HaarBasedMethods.get_trained_model(images_path, labels_path)
        joblib.dump(model, 'haar_trained.pkl')
        print("Модель, основанная на каскадах Хаара, сериализована")
    else:
        model = joblib.load('haar_trained.pkl')
        print("Модель десериализована")

    HaarBasedMethods.detect_and_draw("C:\\Users\\ilyab\\Downloads\\testImage2.jpg",
                               model, "C:\\Users\\ilyab\\PycharmProjects\\kursach\\ProcessedImages\\testImageHaar.jpg")
    print("Обнаружены лица на тестовом изображении")

def demonstrate_hog():
    images_path, labels_path = DataReduction.reduct_images_dataset(
        "C:\\Users\\ilyab\\Downloads\\wider_face\\images",
        "C:\\Users\\ilyab\\Downloads\\wider_face\\labels", 100)

    if not os.path.isfile('hog_trained.pkl'):
        model = HogMethods.get_trained_model(images_path, labels_path)
        joblib.dump(model, 'hog_trained.pkl')
        print("Модель HOG сериализована")
    else:
        model = joblib.load('hog_trained.pkl')
        print("Модель Hog десериализована")

    HogMethods.detect_and_draw("C:\\Users\\ilyab\\Downloads\\testImage2.jpg",
                               model, "C:\\Users\\ilyab\\PycharmProjects\\kursach\\ProcessedImages\\testImageHog.jpg")
    print("Обнаружены лица на тестовом изображении")



if __name__ == '__main__':
    demonstrate_hog()
    demonstrate_haar()