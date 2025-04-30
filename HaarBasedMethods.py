import cv2
from skimage.feature import haar_like_feature
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from torch.utils.data import DataLoader

from GeneralMethods import sliding_window
from GeneralMethods import load_yolo_annotations
from GeneralMethods import draw_detections
from GeneralMethods import FaceDataset

# Извлечь вектор признаков Хаара
def extract_haar_features(image, feature_types=None):

    # Тип проверяемых признаков(2) - горизонтальные и вертикальные прямоугольники
    if feature_types is None:
        feature_types = ['type-2-x']

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (32, 32))
    integral = cv2.integral(gray_image)


    features = []
    for feature_type in feature_types:
        # Извлечение признаков Хаара соответствующих типов в виде вектора
        feats = haar_like_feature(integral, feature_type=feature_type, height=integral.shape[0], width=integral.shape[1], r=0, c=0)
        features.extend(feats)

    return features

# Подготовить данные для обучения
def prepare_data(images, annotations):
    X = []
    y = []


    dataset = FaceDataset(images, annotations, extract_haar_features)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=3, drop_last=False)

    i = 0
    for features, labels in dataloader:

        i += 1
        # Превращаем данные из тензоров в списки и числа
        features1 = features.flatten().tolist()
        labels1 = labels.item()
        X.append(features1)
        y.append(labels1)

        print("Готово " + str(i))

    return np.array(X), np.array(y)
    # i = 0
    # for img_path, annotation in zip(images, annotations):
    #     # Переделываем тензоры обратно в списки
    #     # img_path = img_path[0]
    #     # annotation = [[tensor.item() for tensor in box] for box in annotation]
    #
    #     i += 1
    #     print(img_path + " готово(" + str(i) + ")")
    #     image = cv2.imread(img_path)
    #     if image is None:
    #         print(f"Ошибка загрузки изображения: {img_path}. Пропускаем это изображение.")
    #         continue  # Пропускаем это изображение, если оно не загружено
    #
    #     height, width = image.shape[:2]
    #     # Применение аннотаций
    #     for box in annotation:
    #         x1, y1, x2, y2 = box  # Координаты прямоугольника
    #
    #         face = image[y1:y2, x1:x2]
    #         if face.size > 0:
    #             features = extract_haar_features(face).astype(np.float32)
    #
    #             X.append(features)
    #             y.append(1)  # Лицо
    #
    #
    #     # Генерация негативных примеров
    #     for _ in range(len(annotation)):  # Количество негативных примеров
    #         # Случайный размер окна
    #         window_width = np.random.randint(32, image.shape[1])  # Измените диапазон по необходимости
    #         window_height = np.random.randint(32, image.shape[0])
    #
    #         # Случайные координаты
    #         x1 = np.random.randint(0, width - window_width)
    #         y1 = np.random.randint(0, height - window_height)
    #
    #         not_face = image[y1:y1 + window_height, x1:x1 + window_width]
    #         if not_face.size > 0:
    #             features = extract_haar_features(not_face).astype(np.float32)
    #             X.append(features)
    #             y.append(0)  # Не лицо
    #
    # return np.array(X), np.array(y)


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

    # Обучение модели SVM
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