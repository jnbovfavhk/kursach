import os

import cv2
from torch.utils.data import Dataset
import numpy as np
from skimage.feature import haar_like_feature
import torch



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


# Метод скользящего окна
def sliding_window(image, step_size=None, min_window_size=None, max_window_size=None, aspect_ratio=(1, 2)):
    if max_window_size is None:
        max_window_size = (
            image.shape[1], image.shape[0])  # Установить максимальный размер окна равным размеру изображения

    if min_window_size is None:
        min_window_size = (image.shape[1] // 10, image.shape[0] // 10)

    # if step_size is None:
    #     step_size = min_window_size[0]

    for window_height in range(min_window_size[1], max_window_size[1] + 1,
                               max_window_size[1] // 20):  # Увеличиваем высоту окна
        for window_width in range(min_window_size[0], max_window_size[0] + 1,
                                  max_window_size[0] // 20):  # Увеличиваем ширину окна

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

# Возвращает Intersection over Union (IoU) — метрику пересечения двух прямоугольников
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0

# Генерирует случайный прямоугольник заданных размеров, который не сильно пересекается с уже известными прямоугольниками
def generate_random_box(img_width, img_height, box_width, box_height, exclude_boxes, max_attempts=50):
    for _ in range(max_attempts):
        x1 = np.random.randint(0, img_width - box_width)
        y1 = np.random.randint(0, img_height - box_height)
        x2 = x1 + box_width
        y2 = y1 + box_height
        candidate = [x1, y1, x2, y2]
        if all(iou(candidate, ex) < 0.1 for ex in exclude_boxes):
            return candidate
    return None

# Вспомогательный датасет, который ускоряет обучение
class FaceDataset(Dataset):
    def __init__(self, image_paths, face_boxes_list, extract_features_func):
        self.data = []
        self.extract_features_func = extract_features_func

        for img_path, face_boxes in zip(image_paths, face_boxes_list):
            # Для каждого лица создаём пару: (img_path, box, label=1)
            for box in face_boxes:
                self.data.append((img_path, box, 1))
            # И для каждого лица — не-лицевой пример того же размера с label=0
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            for box in face_boxes:
                box_w = box[2] - box[0]
                box_h = box[3] - box[1]
                nf_box = generate_random_box(w, h, box_w, box_h, face_boxes)
                if nf_box is not None:
                    self.data.append((img_path, nf_box, 0))

        print("FaceDataset инициализирован. длина массива данных: " + str(len(self.data)))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, box, label = self.data[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Не удалось загрузить изображение {img_path}")
        x1, y1, x2, y2 = box
        roi = img[y1:y2, x1:x2]
        print("getitem сработал. Индекс: " + str(idx))
        features = self.extract_features_func(roi)
        # Возвращаем тензор и метку
        return features, label


        # image = cv2.imread(img_path)
        # if image is None:
        #     raise ValueError(f"Ошибка загрузки изображения: {img_path}")
        #
        #
        #
        #
        #
        #
        # x1, y1, x2, y2 = box  # Координаты прямоугольника
        # face = image[y1:y2, x1:x2]
        #
        # face_features = extract_haar_features(face).astype(np.float32)
        # features = face_features
        # if idx < len(self.images) / 2:
        #     label = 1
        # else:
        #     label = 0
        #
        #
        #
        # print(img_path + " готово(" + str(idx) + ")")
        # return np.array(features), np.array(label)


