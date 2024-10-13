import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from PIL import Image, ImageDraw, ImageFont


class Detector:
    def __init__(self) -> None:
        self.yolo_segm = YOLO("./best.pt")
        self.yolo_pixel = YOLO(
            "/home/bob/Desktop/hack/runs/detect/train/weights/best.pt"
        )
        self.yolo_bolt = YOLO(
            "/home/bob/Desktop/hack/runs/detect/boltiks300/weights/best.pt"
        )
        self.yolo_scra4 = YOLO(
            "/home/bob/Desktop/hack/runs/detect/train2/weights/best.pt"
        )
        self.results = []

    def _do_segm(self, img: Image.Image):
        img = img.resize((640, 640))
        results = self.yolo_segm.predict(
            img, conf=0.8, iou=0.1, show_labels=True, show_conf=True, imgsz=640
        )
        img = np.asarray(img, dtype=np.uint8)
        if results[0].masks is not None:
            masks = results[0].masks
            monirot_mask = masks.data.squeeze().cpu().numpy()[0]
            monirot_mask = (monirot_mask * 255).astype(np.uint8)
            monirot_img = cv2.bitwise_and(img, img, mask=monirot_mask)

            if len(results[0].masks)>1:
                laptop_mask = masks.data.squeeze().cpu().numpy()[1]
                laptop_mask = (laptop_mask * 255).astype(np.uint8)
                laptop_img = cv2.bitwise_and(img, img, mask=laptop_mask)
            else:
                laptop_img = img
        else:
            monirot_img = img
            laptop_img = img

        return Image.fromarray(laptop_img), Image.fromarray(monirot_img)

    def detect(self, imgs: Image.Image):
        _results = []
        conf_threshold = 0.2
        iou_threshold = 0.2
        for img in imgs:
            laptop_img, monirot_img = self._do_segm(img)

            bolt_results = self.yolo_bolt.predict(
                source=laptop_img,
                conf=conf_threshold,
                iou=iou_threshold,
                show_labels=True,
                show_conf=True,
                imgsz=640,
            )
            scra4_results = self.yolo_scra4.predict(
                source=laptop_img,
                conf=conf_threshold,
                iou=iou_threshold,
                show_labels=True,
                show_conf=True,
                imgsz=640,
            )
            pixels_results = self.yolo_pixel.predict(
                source=monirot_img,
                conf=conf_threshold,
                iou=iou_threshold,
                show_labels=True,
                show_conf=True,
                imgsz=640,
            )
            for xy, c, cf, xyn in zip(
                bolt_results[0].boxes.xyxy,
                bolt_results[0].boxes.cls,
                bolt_results[0].boxes.conf,
                bolt_results[0].boxes.xywhn,
            ):

                _results.append(
                    [
                        xy.cpu().numpy(),
                        c.cpu().numpy(),
                        cf.cpu().numpy(),
                        xyn.cpu().numpy(),
                        bolt_results[0].names[int(c.cpu().numpy())],
                    ]
                )
            for xy, c, cf, xyn in zip(
                scra4_results[0].boxes.xyxy,
                scra4_results[0].boxes.cls,
                scra4_results[0].boxes.conf,
                scra4_results[0].boxes.xywhn,
            ):

                _results.append(
                    [
                        xy.cpu().numpy(),
                        c.cpu().numpy(),
                        cf.cpu().numpy(),
                        xyn.cpu().numpy(),
                        scra4_results[0].names[int(c.cpu().numpy())],
                    ]
                )
            for xy, c, cf, xyn in zip(
                pixels_results[0].boxes.xyxy,
                pixels_results[0].boxes.cls,
                pixels_results[0].boxes.conf,
                pixels_results[0].boxes.xywhn,
            ):

                _results.append(
                    [
                        xy.cpu().numpy(),
                        c.cpu().numpy(),
                        cf.cpu().numpy(),
                        xyn.cpu().numpy(),
                        pixels_results[0].names[int(c.cpu().numpy())],
                    ]
                )
        self.results.append(_results)
        return bolt_results, scra4_results, pixels_results

    def _plot(self, img, bolt_results, scra4_results, pixels_results):
        bolt_results[0].orig_img = img
        img = bolt_results[0].plot()
        scra4_results[0].orig_img = img
        img = scra4_results[0].plot()
        pixels_results[0].orig_img = img
        img = pixels_results[0].plot()
        return img

    def plot(self, imgs):
        # pil_img = Image.fromarray(img)
        i = 0
        _imgs = []
        for img_i,img in enumerate(imgs):
            if img_i>=len(self.results):
                result = []
            else:
                result = self.results[img_i]
            for xy, c, cf, xyn, name in result:
                ImageDraw.Draw(img).rectangle(xy,outline ="red",width=2)
                ImageDraw.Draw(img).text(
                    (int(xy[0]), int(xy[1]-40)),
                    f"{name} id:{i}",
                    (255, 0, 0),
                    font=ImageFont.load_default(size=30),
                )
                i+=1
            _imgs.append(img)
        return _imgs

    def delete_bbox_by_id(self, id: int):
        self.results.pop(id)

    # def parse_res(self, bolt_results, scra4_results, pixels_results):
