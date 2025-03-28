import os

import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Функция для извлечения признаков HOG
def extract_hog_features(image):
    image = cv2.resize(image, (64, 128))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features


# def prepare_data(images, annotations):
# X = []
# y = []
#
# for img_path, annotation in zip(images, annotations):
#     print(img_path + " готово")
#     image = cv2.imread(img_path)
#     if image is None:
#         print(f"Ошибка загрузки изображения: {img_path}. Пропускаем это изображение.")
#         continue  # Пропускаем это изображение, если оно не загружено
#
#     height, width = image.shape[:2]
#     # Применение аннотаций YOLO
#     for box in annotation:
#         x_center, y_center, w, h = box  # Координаты из YOLO
#         x1 = int((x_center - w / 2) * width)
#         y1 = int((y_center - h / 2) * height)
#         x2 = int((x_center + w / 2) * width)
#         y2 = int((y_center + h / 2) * height)
#         face = image[y1:y2, x1:x2]
#         if face.size > 0:
#             features = extract_hog_features(face).astype(float32)
#             X.append(features)
#             y.append(1)  # Лицо
#
#     # Генерация негативных примеров (например, случайные окна)
#     for _ in range(100):  # Количество негативных примеров
#         x1 = np.random.randint(0, width - 64)
#         y1 = np.random.randint(0, height - 64)
#         not_face = image[y1:y1 + 64, x1:x1 + 64]
#         if not_face.size > 0:
#             features = extract_hog_features(not_face).astype(float32)
#             X.append(features)
#             y.append(0)  # Не лицо
#
# return np.array(X), np.array(y)

# Подготовка данных(X - hog векторы лиц или не лиц, y = 1, если x - лицо, 0 - иначе
def prepare_data(images, annotations, batch_size=10):
    X = np.array([])
    y = np.array([])

    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_annotations = annotations[i:i + batch_size]

        for img_path, annotation in zip(batch_images, batch_annotations):
            print(img_path + " готово")
            image = cv2.imread(img_path)

            if image is None:
                print(f"Ошибка загрузки изображения: {img_path}. Пропускаем это изображение.")
                continue  # Пропускаем это изображение, если оно не загружено

            height, width = image.shape[:2]
            # Применение аннотаций YOLO
            for box in annotation:
                x_center, y_center, w, h = box  # Координаты из YOLO
                x1 = int((x_center - w / 2) * width)
                y1 = int((y_center - h / 2) * height)
                x2 = int((x_center + w / 2) * width)
                y2 = int((y_center + h / 2) * height)
                face = image[y1:y2, x1:x2]
                if face.size > 0:
                    features = extract_hog_features(face)
                    X = np.append(X, features)
                    y = np.append(y, 1)  # Лицо

            # Генерация негативных примеров (например, случайные окна)
            for _ in range(100):  # Количество негативных примеров
                x1 = np.random.randint(0, width - 64)
                y1 = np.random.randint(0, height - 64)
                not_face = image[y1:y1 + 64, x1:x1 + 64]
                if not_face.size > 0:
                    features = extract_hog_features(not_face)
                    X = np.append(X, features)
                    y = np.append(y, 0)  # Не лицо

        # Преобразуем в numpy массивы после обработки батча
        if len(X) > 0 and len(y) > 0:
            X = np.array(X)
            y = np.array(y)
            print(f"Обработано {len(X)} примеров.")

    return np.array(X), np.array(y)


# Обучение модели SVM
def train_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)

    return model


# Метод скользящего окна
def sliding_window(image, step_size=8, window_size=(64, 64)):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


# Обнаружение лиц с помощью скользящего окна
def detect_faces(image, model):
    detections = []
    for (x, y, window) in sliding_window(image):
        if window.shape[0] != 64 or window.shape[1] != 64:
            continue
        features = extract_hog_features(window)
        prediction = model.predict([features])
        if prediction == 1:  # Если предсказано, что это лицо
            detections.append((x, y, x + 64, y + 64))  # Записываем координаты окна

    return detections

# Отрисовка прямоугольников на изображении
def draw_detections(image, detections):
    for (x1, y1, x2, y2) in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Рисуем прямоугольник
    return image


# Возвращает список полных путей изображений и к каждому из них список координат лиц
def load_yolo_annotations(images_dir, annotations_dir):
    annotations = []
    image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

    i = 0
    for image_file in image_files:
        i += 1
        print(i)

        # Получение размера изображения
        image = cv2.imread(image_file)

        height, width = image.shape[:2]

        # Путь к соответствующему файлу аннотаций
        annotation_file = os.path.basename(image_file).rstrip('.jpg').rstrip('.png') + '.txt'
        annotation_path = os.path.join(annotations_dir, annotation_file)

        if os.path.exists(annotation_path):
            image_annotations = []
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    # Разделение строки на элементы
                    parts = line.strip().split()
                    class_id = int(parts[0])  # ID класса
                    x_center = float(parts[1]) * width  # Координата x центра
                    y_center = float(parts[2]) * height  # Координата y центра
                    w = float(parts[3]) * width  # Ширина
                    h = float(parts[4]) * height  # Высота

                    # Рассчитываем координаты прямоугольника
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)

                    image_annotations.append([x1, y1, x2, y2])  # Добавляем координаты

            annotations.append(image_annotations)  # Добавляем аннотации для изображения

    return image_files, annotations

def get_trained_model():
    # Загрузка изображений и аннотаций
    images, annotations = load_yolo_annotations("C:\\Users\\ilyab\\Downloads\\wider_face\\images", "C:\\Users\\ilyab\\Downloads\\wider_face\\labels")

    # Подготовка данных
    X, y = prepare_data(images, annotations)

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

# # Показ результата
# plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
