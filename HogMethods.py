import os

import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split


# Функция для извлечения признаков HOG
def extract_hog_features(image):
    image = cv2.resize(image, (64, 128))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
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
                features = extract_hog_features(face).astype(np.float32)
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
                features = extract_hog_features(not_face).astype(np.float32)
                X.append(features)
                y.append(0)  # Не лицо

    return np.array(X), np.array(y)


# Обучение модели SVM
def train_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = svm.SVC(kernel='rbf')
    model.fit(X_train, y_train)

    return model



# Метод скользящего окна
def sliding_window(image, step_size=None, min_window_size=None, max_window_size=None, aspect_ratio=(1, 2)):
    if max_window_size is None:
        max_window_size = (image.shape[1], image.shape[0]) # Установить максимальный размер окна равным размеру изображения

    if min_window_size is None:
        min_window_size = (image.shape[1] // 20, image.shape[0] // 20)

    if step_size is None:
        step_size = min_window_size[0]

    for window_height in range(min_window_size[1], max_window_size[1] + 1, max_window_size[1] // 20):  # Увеличиваем высоту окна
        for window_width in range(min_window_size[0], max_window_size[0] + 1, max_window_size[0] // 20):  # Увеличиваем ширину окна

            # Находим соотношение сторон
            width_to_height_ratio = window_width / window_height
            height_to_width_ratio = window_height / window_width

            step_size = window_width
            # Проверяем, что хотя бы одно из соотношений в заданном интервале(где может находится лицо)
            if (aspect_ratio[0] <= width_to_height_ratio <= aspect_ratio[1]) or \
                    (aspect_ratio[0] <= height_to_width_ratio <= aspect_ratio[1]):
                for y in range(0, image.shape[0] - window_height + 1, step_size):
                    for x in range(0, image.shape[1] - window_width + 1, step_size):
                        print(f'x = {x}, y = {y}, width = {window_width}, height = {window_height}')
                        yield (x, y, image[y:y + window_height, x:x + window_width])


# Обнаружение лиц с помощью скользящего окна
def detect_faces(image, model):
    detections = []
    for (x, y, window) in sliding_window(image):
        features = extract_hog_features(window)
        prediction = model.predict([features])
        if prediction == 1:  # Если предсказано, что это лицо
            detections.append((x, y, x + window.shape[1], y + window.shape[0]))  # Записываем координаты окна

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

def get_trained_model(images_path, labels_path):
    # Загрузка изображений и аннотаций
    images, annotations = load_yolo_annotations(images_path, labels_path)

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
