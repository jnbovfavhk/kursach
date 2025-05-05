import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split

from GeneralMethods import load_yolo_annotations
from GeneralMethods import draw_detections
from GeneralMethods import sliding_window
from GeneralMethods import FaceDataset


# Функция для извлечения признаков HOG
def extract_hog_features(image):
    if image is None or image.size == 0:
        return np.zeros(8100)
    image = cv2.resize(image, (128, 128))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features




def extract_hog_features_batch(images):
    features_list = []
    for img in images:
        features = extract_hog_features(img)
        features_list.append(features)

    return features_list


# Возвращает X: вектор HOG, y: 1, если это лицо, 0 если нет. Генерирует не-лица
def prepare_data(images, annotations, batch_size=32):
    X = []
    y = []


    dataset = FaceDataset(images, annotations, extract_hog_features)

    # batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    def process_item(idx):
        features, label = dataset[idx]
        print(features)
        return features, label

    i = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        print("Готово - " + str(i))
        indices = list(range(len(dataset)))
        results = executor.map(process_item, indices)
        for features, label in results:
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


# Обучение модели SVM
def train_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = svm.SVC(kernel='rbf', gamma='scale')
    model.fit(X_train, y_train)

    return model



# Обнаружение лиц с помощью скользящего окна
def detect_faces(image, model):
    detections = []
    for (x, y, window) in sliding_window(image):
        features = extract_hog_features(window)
        prediction = model.predict([features])
        if prediction == 1:  # Если предсказано, что это лицо
            detections.append((x, y, x + window.shape[1], y + window.shape[0]))  # Записываем координаты окна

    return detections



def get_trained_model(images_path, labels_path):
    # Загрузка изображений и аннотаций
    images, annotations = load_yolo_annotations(images_path, labels_path)

    # Подготовка данных
    X, y = prepare_data(images, annotations)

    print("Идет обучение...")
    # Обучение модели SVM
    model = train_svm(X, y)
    return model


def detect_and_draw(path, model, path_to_save):
    # Обнаружение лиц на новом изображении
    test_image = cv2.imread(path)
    detections = detect_faces(test_image, model)
    print(detections)
    # Отрисовка обнаруженных лиц
    output_image = draw_detections(test_image, detections)
    cv2.imwrite(path_to_save, output_image)
