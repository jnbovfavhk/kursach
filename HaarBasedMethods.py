import cv2
from skimage.feature import haar_like_feature
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from GeneralMethods import sliding_window
from GeneralMethods import load_yolo_annotations
from GeneralMethods import draw_detections

def extract_haar_features(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Извлечение признаков Хаара в виде вектора. Тип проверяемых признаков(2) - горизонтальные прямоугольники
    features = haar_like_feature(gray_image, feature_type='type-2-x')
    return features

def prepare_data(images, annotations):
    X = []
    y = []

    i = 0
    for img_path, annotation in zip(images, annotations):
        i += 1
        print(img_path + " готово(" + str(i) + ")")
        image = cv2.imread(img_path)
        if image is None:
            print(f"Ошибка загрузки изображения: {img_path}. Пропускаем это изображение.")
            continue  # Пропускаем это изображение, если оно не загружено

        height, width = image.shape[:2]
        # Применение аннотаций YOLO
        for box in annotation:
            x1, y1, x2, y2 = box  # Координаты прямоугольника

            face = image[y1:y2, x1:x2]
            if face.size > 0:
                features = extract_haar_features(face).astype(np.float32)
                X.append(features)
                y.append(1)  # Лицо

        # Генерация негативных примеров
        for _ in range(100):  # Количество негативных примеров
            # Случайный размер окна
            window_width = np.random.randint(32, 128)  # Измените диапазон по необходимости
            window_height = np.random.randint(32, 128)

            # Случайные координаты
            x1 = np.random.randint(0, width - window_width)
            y1 = np.random.randint(0, height - window_height)

            not_face = image[y1:y1 + window_height, x1:x1 + window_width]
            if not_face.size > 0:
                features = extract_haar_features(not_face).astype(np.float32)
                X.append(features)
                y.append(0)  # Не лицо

    return np.array(X), np.array(y)


def train_adaboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    adaboost_classifier = AdaBoostClassifier()
    adaboost_classifier.fit(X_train, y_train)

    return adaboost_classifier


# Обнаружение лиц с помощью скользящего окна
def detect_faces(image, model):
    detections = []
    for (x, y, window) in sliding_window(image):
        features = extract_haar_features(window)
        prediction = model.predict([features])
        if prediction == 1:  # Если предсказано, что это лицо
            detections.append((x, y, x + window.shape[1], y + window.shape[0]))  # Записываем координаты окна

    return detections


def get_trained_model(images_path, labels_path):
    # Загрузка изображений и аннотаций
    images, annotations = load_yolo_annotations(images_path, labels_path)

    # Подготовка данных
    X, y = prepare_data(images, annotations)

    # Обучение модели SVM
    model = train_adaboost(X, y)
    return model

def detect_and_draw(path, model, path_to_save):
    # Обнаружение лиц на новом изображении
    test_image = cv2.imread(path)
    detections = detect_faces(test_image, model)
    print(detections)
    # Отрисовка обнаруженных лиц
    output_image = draw_detections(test_image, detections)
    cv2.imwrite(path_to_save, output_image)