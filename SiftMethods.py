import os

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from GeneralMethods import load_yolo_annotations, sliding_window, draw_detections


class FaceDetector:
    def __init__(self, n_clusters=1000, svm_kernel='linear'):
        """
        Инициализация детектора лиц.

        Параметры:
            n_clusters: количество кластеров для K-means
            svm_kernel: тип ядра для SVM ('linear', 'rbf', etc.)
        """
        self.n_clusters = n_clusters
        self.svm_kernel = svm_kernel
        self.sift = cv2.SIFT.create()
        self.kmeans = None
        self.svm = None
        self.scaler = StandardScaler()
        self.pipeline = None

    def _extract_sift_descriptors(self, image):
        """Извлекает SIFT-дескрипторы из изображения"""
        kp, desc = self.sift.detectAndCompute(image, None)
        return desc if desc is not None else np.array([])

    def _create_bovw_vector(self, descriptors):
        """Преобразует дескрипторы в BoVW-вектор"""
        if len(descriptors) == 0:
            return np.zeros(self.n_clusters)

        labels = self.kmeans.predict(descriptors)
        hist = np.bincount(labels, minlength=self.n_clusters)
        return hist / (np.linalg.norm(hist) + 1e-6)

#  Собирает дескрипторы для обучения K-means и примеры для классификации
    # Возвращает:
    # all_descriptors: все дескрипторы для обучения K-means
    # samples: список кортежей (descriptors, label) для обучения модели
    def _collect_training_samples(self, image_paths, annotations):

        all_descriptors = []
        samples = []

        for img_path, rects in zip(image_paths, annotations):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            height, width = img.shape

            # Положительные примеры (лица)
            for (x1, y1, x2, y2) in rects:
                face_roi = img[y1:y2, x1:x2]
                if face_roi.size > 0:
                    desc = self._extract_sift_descriptors(face_roi)
                    if len(desc) > 0:
                        all_descriptors.extend(desc)
                        samples.append((desc, 1))

            # Негативные примеры (фон)
            for _ in range(len(rects)):
                while True:
                    w, h = np.random.randint(50, 150, 2)
                    x = np.random.randint(0, width - w)
                    y = np.random.randint(0, height - h)

                    overlap = False
                    for (fx1, fy1, fx2, fy2) in rects:
                        if (x < fx2 and x + w > fx1 and
                                y < fy2 and y + h > fy1):
                            overlap = True
                            break

                    if not overlap:
                        bg_roi = img[y:y + h, x:x + w]
                        if bg_roi.size > 0:
                            desc = self._extract_sift_descriptors(bg_roi)
                            if len(desc) > 0:
                                all_descriptors.extend(desc)
                                samples.append((desc, 0))
                        break

        return np.vstack(all_descriptors), samples

    def train(self, image_paths, annotations):
        """
        Полный цикл обучения:
        1. Извлечение дескрипторов
        2. Обучение K-means
        3. Создание BoVW-векторов
        4. Обучение SVM
        """
        # 1. Сбор всех дескрипторов и примеров
        all_descriptors, samples = self._collect_training_samples(image_paths, annotations)

        # 2. Обучение K-means
        print("Обучение K-means...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        self.kmeans.fit(all_descriptors)

        # 3. Создание BoVW-векторов
        print("Создание BoVW-векторов...")
        X = []
        y = []

        for desc, label in samples:
            bovw = self._create_bovw_vector(desc)
            X.append(bovw)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        # 4. Обучение SVM
        print("Обучение SVM...")
        self.svm = SVC(kernel=self.svm_kernel,
                       class_weight='balanced',
                       probability=True)

        # Создание pipeline с нормализацией
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', self.svm)
        ])

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Обучение
        self.pipeline.fit(X_train, y_train)

        # Оценка
        print("Оценка модели:")
        y_pred = self.pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))

        return self

    def predict_image(self, image_path, threshold=0.8):
        """
        Предсказание для одного изображения.

        Параметры:
            image_path: путь к изображению
            threshold: порог уверенности

        Возвращает:
            True/False - есть ли лицо на изображении
            confidence - уверенность модели
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, 0.0

        desc = self._extract_sift_descriptors(img)
        if len(desc) == 0:
            return False, 0.0

        bovw = self._create_bovw_vector(desc).reshape(1, -1)
        proba = self.pipeline.predict_proba(bovw)[0][1]

        return proba >= threshold, proba

    def save_model(self, path):
        """Сохраняет модель на диск"""
        joblib.dump({
            'kmeans': self.kmeans,
            'pipeline': self.pipeline,
            'n_clusters': self.n_clusters,
            'svm_kernel': self.svm_kernel
        }, path)

    @classmethod
    def load_model(cls, path):
        """Загружает модель с диска"""
        data = joblib.load(path)
        detector = cls(n_clusters=data['n_clusters'],
                       svm_kernel=data['svm_kernel'])
        detector.kmeans = data['kmeans']
        detector.pipeline = data['pipeline']
        return detector

    def _apply_nms(self, rectangles, min_neighbors):
        """Применяет Non-Maximum Suppression к списку прямоугольников"""
        if len(rectangles) == 0:
            return []

        # Сортируем прямоугольники по уверенности (по убыванию)
        rectangles.sort(key=lambda x: x[4], reverse=True)

        final_rectangles = []
        suppressed = [False] * len(rectangles)

        for i in range(len(rectangles)):
            if suppressed[i]:
                continue

            current = rectangles[i]
            final_rectangles.append(current[:4])
            neighbors = 0

            for j in range(i + 1, len(rectangles)):
                if suppressed[j]:
                    continue

                # Вычисляем IoU (Intersection over Union)
                x1_i, y1_i, x2_i, y2_i = current[:4]
                x1_j, y1_j, x2_j, y2_j = rectangles[j][:4]

                xi1 = max(x1_i, x1_j)
                yi1 = max(y1_i, y1_j)
                xi2 = min(x2_i, x2_j)
                yi2 = min(y2_i, y2_j)

                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                iou = inter_area / ((x2_i - x1_i) * (y2_i - y1_i) + (x2_j - x1_j) * (y2_j - y1_j) - inter_area)

                if iou > 0.3:  # Порог перекрытия
                    neighbors += 1
                    if neighbors >= min_neighbors:
                        suppressed[j] = True

        return final_rectangles

    def detect_faces(self, image_path, threshold = 0.8, min_neighbors = 1):
        """
        Обнаружение лиц на изображении с помощью вашего sliding_window

        Параметры:
            image_path: путь к изображению
            threshold: порог вероятности для классификации как лицо
            min_neighbors: минимальное количество соседей для NMS
            min_window_size: минимальный размер окна
            max_window_size: максимальный размер окна
            aspect_ratio: допустимое соотношение сторон (min, max)

        Возвращает:
            Список прямоугольников лиц в формате (x1, y1, x2, y2)
        """
        img = cv2.imread(image_path)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rectangles = []

        # Используем ваш sliding_window
        for (x, y, window) in sliding_window(gray):
            desc = self._extract_sift_descriptors(window)
            if len(desc) == 0:
                continue

            bovw = self._create_bovw_vector(desc).reshape(1, -1)
            proba = self.pipeline.predict_proba(bovw)[0][1]

            if proba >= threshold:
                win_h, win_w = window.shape[:2]
                rectangles.append((x, y, x + win_w, y + win_h, proba))

        # Применяем Non-Maximum Suppression
        return self._apply_nms(rectangles, min_neighbors)

    def detect_and_draw(self, img_path, path_to_save, threshold = 0.7, min_neighbors = 1):
        """
        Обнаруживает лица на изображении, рисует прямоугольники и сохраняет результат
        Возвращает:
            bool: True если обнаружение и сохранение прошло успешно, False в противном случае
        """
        try:
            # 1. Загрузка изображения
            img = cv2.imread(img_path)
            if img is None:
                print(f"Ошибка: не удалось загрузить изображение {img_path}")
                return False

            # 2. Обнаружение лиц
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rectangles = []

            for (x, y, window) in sliding_window(gray):
                desc = self._extract_sift_descriptors(window)
                if len(desc) == 0:
                    continue

                bovw = self._create_bovw_vector(desc).reshape(1, -1)
                proba = self.pipeline.predict_proba(bovw)[0][1]

                if proba >= threshold:
                    win_h, win_w = window.shape[:2]
                    rectangles.append((x, y, x + win_w, y + win_h, proba))

            # 3. Применяем Non-Maximum Suppression
            final_detections = self._apply_nms(rectangles, min_neighbors)

            # 4. Рисуем прямоугольники и сохраняем
            result_img = draw_detections(img, final_detections)

            # Создаем директорию, если ее нет
            os.makedirs(os.path.dirname(path_to_save) or os.path.dirname(path_to_save), exist_ok=True)

            # 5. Сохраняем результат
            if not cv2.imwrite(path_to_save, result_img):
                print(f"Ошибка: не удалось сохранить изображение {path_to_save}")
                return False

            print(f"Успешно! Обнаружено лиц: {len(final_detections)}. Результат сохранен в {path_to_save}")
            return True

        except Exception as e:
            print(f"Ошибка в detect_and_draw: {str(e)}")
            return False



def get_trained_model(images_path, labels_path):
    img_paths, annotations = load_yolo_annotations(images_path, labels_path)
    model = FaceDetector()
    model.train(img_paths, annotations)
    return model