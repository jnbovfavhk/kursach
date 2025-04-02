import HogMethods
import joblib
import os
import DataReduction

if __name__ == '__main__':
    images_path, labels_path = DataReduction.reduct_images_dataset("C:\\Users\\ilyab\\Downloads\\wider_face\\images",
                                        "C:\\Users\\ilyab\\Downloads\\wider_face\\labels")

    if not os.path.isfile('hog_trained.pkl'):
        model = HogMethods.get_trained_model(images_path, labels_path)
        joblib.dump(model, 'hog_trained.pkl')
        print("Модель сериализована")
    else:
        model = joblib.load('hog_trained.pkl')
        print("Модель десериализована")

    HogMethods.detect_and_draw("C:\\Users\\ilyab\\Downloads\\testImage2.jpg",
                               model, "C:\\Users\\ilyab\\PycharmProjects\\kursach\\ProcessedImages\\testImage.jpg")
    print("Обнаружены лица на тестовом изображении")
