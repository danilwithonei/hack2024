import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import time

class Detector:
    def __init__(self) -> None:
        """Инициализация детектора с моделями YOLO."""
        self.yolo_segm = YOLO("weights/yolo_segm.pt")  # YOLO для сегментации
        self.yolo_pixel = YOLO("weights/yolo_pixel.pt")  # YOLO для обнаружения битых пикселей
        self.yolo_bolt = YOLO("weights/yolo_bolt.pt")  # YOLO для обнаружения болтов
        self.yolo_scra4 = YOLO("weights/yolo_scra4.pt")  # YOLO для обнаружения царапин
        self.results = []  # Список для хранения результатов

    def _do_segm(self, img: Image.Image):
        """Выполнение сегментации на входное изображение."""
        img = img.resize((640, 640))  # Изменение размера изображения
        results = self.yolo_segm.predict(
            img, conf=0.8, iou=0.1, show_labels=True, show_conf=True, imgsz=640
        )
        img = np.asarray(img, dtype=np.uint8)  # Преобразование изображения в массив NumPy

        # Инициализация изображений для разных объектов
        monirot_img, laptop_img = img.copy(), img.copy()  # По умолчанию оригинальное изображение

        if results[0].masks is not None:
            masks = results[0].masks
            # Извлечение масок для соответствующих объектов
            try:
                monirot_mask = (masks.data.squeeze().cpu().numpy()[0] * 255).astype(np.uint8)
                monirot_img = cv2.bitwise_and(img, img, mask=monirot_mask)
            except:
                monirot_img = img
            

            if len(results[0].masks) > 1:
                laptop_mask = (masks.data.squeeze().cpu().numpy()[1] * 255).astype(np.uint8)
                laptop_img = cv2.bitwise_and(img, img, mask=laptop_mask)

        return Image.fromarray(laptop_img), Image.fromarray(monirot_img)

    def detect(self, imgs: Image.Image):
        """Обнаружение объектов на входных изображениях."""
        conf_threshold = 0.2  # Порог достоверности
        iou_threshold = 0.2  # Порог IoU

        for img in imgs:
            _results = []  # Список для хранения результатов для текущего изображения
            laptop_img, monirot_img = self._do_segm(img)  # Сегментация изображения

            # Предсказание болтов, царапин и битых пикселей
            bolt_results = self.yolo_bolt.predict(source=laptop_img, conf=conf_threshold, iou=iou_threshold)
            scra4_results = self.yolo_scra4.predict(source=laptop_img, conf=conf_threshold, iou=iou_threshold)
            pixels_results = self.yolo_pixel.predict(source=monirot_img, conf=conf_threshold, iou=iou_threshold)

            # Извлечение результатов для болтов
            for xy, c, cf, xyn in zip(bolt_results[0].boxes.xyxy, bolt_results[0].boxes.cls,
                                       bolt_results[0].boxes.conf, bolt_results[0].boxes.xywhn):
                _results.append([
                    xy.cpu().numpy(),
                    c.cpu().numpy(),
                    cf.cpu().numpy(),
                    xyn.cpu().numpy(),
                    bolt_results[0].names[int(c.cpu().numpy())],
                ])

            # Извлечение результатов для царапин
            for xy, c, cf, xyn in zip(scra4_results[0].boxes.xyxy, scra4_results[0].boxes.cls,
                                       scra4_results[0].boxes.conf, scra4_results[0].boxes.xywhn):
                _results.append([
                    xy.cpu().numpy(),
                    c.cpu().numpy(),
                    cf.cpu().numpy(),
                    xyn.cpu().numpy(),
                    scra4_results[0].names[int(c.cpu().numpy())],
                ])

            # Извлечение результатов для битых пикселей
            for xy, c, cf, xyn in zip(pixels_results[0].boxes.xyxy, pixels_results[0].boxes.cls,
                                      pixels_results[0].boxes.conf, pixels_results[0].boxes.xywhn):
                _results.append([
                    xy.cpu().numpy(),
                    c.cpu().numpy(),
                    cf.cpu().numpy(),
                    xyn.cpu().numpy(),
                    pixels_results[0].names[int(c.cpu().numpy())],
                ])

            self.results.append(_results)  # Сохранение результатов для изображения

        return bolt_results, scra4_results, pixels_results

    def _plot(self, img, bolt_results, scra4_results, pixels_results):
        """Отображение рамок на изображении."""
        bolt_results[0].orig_img = img  # Установка оригинального изображения
        img = bolt_results[0].plot()  # Отображение результатов для болтов
        scra4_results[0].orig_img = img
        img = scra4_results[0].plot()  # Отображение результатов для царапин
        pixels_results[0].orig_img = img
        img = pixels_results[0].plot()  # Отображение результатов для битых пикселей
        return img

    def plot(self, imgs):
        """Рисование рамок и меток на изображениях."""
        i = 0
        _imgs = []  # Список для хранения изображений с рамками
        for img_i, img in enumerate(imgs):
            # Получение результатов для текущего изображения
            result = self.results[img_i] if img_i < len(self.results) else []

            for xy, c, cf, xyn, name in result:
                # Рисование рамок и меток на изображении
                ImageDraw.Draw(img).rectangle(xy, outline="red", width=2)
                ImageDraw.Draw(img).text(
                    (int(xy[0]), int(xy[1] - 40)),
                    f"{name} id:{i}",
                    (255, 0, 0),
                    font=ImageFont.load_default(size=30),
                )
                i += 1
            _imgs.append(img)  # Добавление измененного изображения в список
        return _imgs

    def delete_bbox_by_id(self, ids: int):
        """Удаление рамок на основе указанных идентификаторов."""
        del_indexes = []  # Список для хранения индексов удаляемых рамок
        ii = 0
        for i in range(len(self.results)):
            for j in range(len(self.results[i])):
                if str(ii) in ids:  # Проверка, есть ли идентификатор в списке
                    del_indexes.append([i, j])  # Запись индекса для удаления
                ii += 1
        for i, j in del_indexes:
            self.results[i].pop(j)  # Удаление рамки из результатов

    def get_report(self, laptop_id):
        """Генерация отчета на основе результатов обнаружения."""
        res_str = str(self.results)  # Преобразование результатов в строку
        data = {
            "id": laptop_id,
            "date": time.time(),  # Текющая дата и время
            "царапина": int("'c'" in res_str),  # Проверка на наличие царапин
            "скол": int("'s'" in res_str),  # Проверка на наличие сколов
            "замок": int("'bad'" in res_str),  # Проверка на наличие замков
            "отсутсвует болт": int("'empty'" in res_str),  # Проверка на отсутствие болтов
            "битый пиксель": int("'pixel'" in res_str),  # Проверка на битые пиксели
        }
        return data  # Возврат данных отчета
