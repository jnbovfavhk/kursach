import HogMethods
import joblib
import os

if __name__ == '__main__':
    if not os.path.isfile('hog_trained.pkl'):
        model = HogMethods.get_trained_model()
        joblib.dump(model, 'hog_trained.pkl')
    else:
        model = joblib.load('hog_trained.pkl')

    HogMethods.detect_and_draw("C:\\Users\\ilyab\\Downloads\\testImage.jpg",
                               model, "C:\\Users\\ilyab\\PycharmProjects\\kursach\\ProcessedImages\\testImage")
    print("Обнаружены лица на тестовом изображении")
