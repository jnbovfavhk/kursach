from concurrent.futures import ThreadPoolExecutor

import cv2
from skimage.feature import haar_like_feature
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

from GeneralMethods import sliding_window
from GeneralMethods import load_yolo_annotations
from GeneralMethods import draw_detections
from GeneralMethods import FaceDataset
from PIL import Image
import torch


# Извлечь вектор признаков Хаара
def extract_haar_features(image, feature_types=None):
    # Тип проверяемых признаков(2) - горизонтальные и вертикальные прямоугольники
    if feature_types is None:
        feature_types = ['type-2-x']

    if isinstance(image, np.ndarray):
        gray_image = np.array(Image.fromarray(image).convert('L'))
    else:
        gray_image = np.array(image.convert('L'))

    gray_image = np.array(Image.fromarray(gray_image).resize((32, 32)))

    integral = np.cumsum(np.cumsum(gray_image, axis=0), axis=1)

    features = []
    for feature_type in feature_types:
        # Извлечение признаков Хаара соответствующих типов в виде вектора
        feats = haar_like_feature(integral, feature_type=feature_type, height=integral.shape[0], width=integral.shape[1], r=0, c=0)
        features.extend(feats)

    return features


def extract_haar_features_batch(images, feature_types=None):
    if feature_types is None:
        feature_types = ['type-2-x']

    # Обрабатываем все изображения в батче
    integrals = []
    for img in images:
        gray = np.array(Image.fromarray(img).convert('L').resize((32, 32)))
        integrals.append(np.cumsum(np.cumsum(gray, axis=0), axis=1))

    # Векторизованное извлечение признаков
    all_features = []
    for integral in integrals:
        features = []
        for feature_type in feature_types:
            feats = haar_like_feature(integral, feature_type=feature_type,
                                      height=32, width=32, r=0, c=0)
            features.extend(feats)
        all_features.append(features)

    return all_features


# Подготовить данные для обучения
def prepare_data(images, annotations, batch_size=32):
    X = []
    y = []
    dataset = FaceDataset(images, annotations, extract_haar_features_batch)

    # # Функция для обработки батча
    # def process_batch(indices):
    #     batch_features, batch_labels = [], []
    #     for idx in indices:
    #         features, label = dataset[idx]
    #         batch_features.append(list(map(int, features)))
    #         batch_labels.append(label)
    #     return batch_features, batch_labels


    indices = list(range(len(dataset)))
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    i = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(dataset.get_batch, batches)

        for features_batch, label_batch in results:
            i += 1
            X.extend(features_batch)
            y.extend(label_batch)
            print("Готово - " + str(i))

    return np.array(X), np.array(y)


# Обучить AdaBoost на X и y
def train_adaboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    adaboost_classifier = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
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

# Возвращает обученную модель
def get_trained_model(images_path, labels_path):
    # Загрузка изображений и аннотаций
    images, annotations = load_yolo_annotations(images_path, labels_path)

    # Подготовка данных
    X, y = prepare_data(images, annotations)

    # Обучение модели AdaBoost
    print("Идет обучение...")
    model = train_adaboost(X, y)
    return model

# Активный метод для обнаружения лиц на новых изображениях и рисования прямоугольных меток на них
def detect_and_draw(path, model, path_to_save):
    # Обнаружение лиц на новом изображении
    test_image = cv2.imread(path)
    detections = detect_faces(test_image, model)
    print(detections)
    # Отрисовка обнаруженных лиц
    output_image = draw_detections(test_image, detections)
    cv2.imwrite(path_to_save, output_image)