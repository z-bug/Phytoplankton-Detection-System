import os
import onnxruntime
import numpy as np
import cv2
import time


class yolo_onnx():
    def __init__(self, onnx_path, clas_names, conf_thres=0.25):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        print(self.ort_session.get_providers())

        self.new_shape = (640, 640)
        self.clas_names = clas_names        # 类别名列表，如 ['1','2',...,'20']
        self.conf_thres = conf_thres        # 置信度阈值，0~1 浮点
        self.color = (114, 114, 114)
        self.color_list = self.SetColor(len(self.clas_names))

    def detect(self, image):
        """
        检测入口
        返回: result, boxes, labels, scores
            result  : list of dict，原始结果，用于 draw_img
            boxes   : list of (x1,y1,x2,y2)，整数坐标
            labels  : list of int，类别索引
            scores  : list of float，置信度 0~1
        """
        img = image.copy()

        # 前处理
        img, pad, scale = self.resize(img)
        float_img = self.normalize(img)

        # 推理
        ort_inputs = {self.ort_session.get_inputs()[0].name: float_img}
        pred = self.ort_session.run(None, ort_inputs)[0]   # (1, 84, 8400)
        pred = pred.transpose(0, 2, 1)                     # (1, 8400, 84)

        # 后处理
        pred = self.non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=0.45, classes=None)[0]
        result = self.pred2result(pred, pad, scale, image.shape)

        boxes, labels, scores = [], [], []
        for box_dict in result:
            clas_id = box_dict["cls_id"]           # int，类别索引
            conf    = box_dict["conf"]             # float，0~1 置信度
            x, y, w, h = box_dict["box"]
            box_xyxy = (x, y, x + w, y + h)
            boxes.append(box_xyxy)
            labels.append(clas_id)
            scores.append(conf)

        # 跨类 NMS / 优先级过滤（priority_delete_class=12）
        boxes, labels, scores = self.process_detections(boxes, labels, scores, iou_threshold=0.5, priority_delete_class=12)

        # 返回顺序固定为: result, boxes, labels, scores
        return result, boxes, labels, scores

    # ------------------------------------------------------------------ #
    #  图像预处理
    # ------------------------------------------------------------------ #
    def resize(self, image):
        shape = image.shape[:2]
        scale = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        img_new = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        dh = self.new_shape[0] - img_new.shape[0]
        dw = self.new_shape[1] - img_new.shape[1]
        left, top = dw // 2, dh // 2
        right, bottom = dw - left, dh - top

        img = cv2.copyMakeBorder(img_new, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)
        return img, [left, top], scale

    def normalize(self, resize_img):
        img = resize_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        img /= 255.0
        if img.ndim == 3:
            img = img[None]
        return img

    # ------------------------------------------------------------------ #
    #  后处理：预测框 → 结果字典列表
    # ------------------------------------------------------------------ #
    def pred2result(self, pred, pad, scale, img_shape):
        det = pred
        result_list = []
        if len(det):
            det[:, :4] = self.scale_coords(det[:, :4], pad, scale, img_shape)
            for *xyxy, conf, cls in reversed(det):
                cls  = int(cls)
                conf = float(round(conf, 4))        # 保留4位，0~1 浮点
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                w  = int(xyxy[2] - xyxy[0])
                h  = int(xyxy[3] - xyxy[1])
                result_list.append({
                    "cls_id": cls,
                    "conf":   conf,
                    "box":    [x1, y1, w, h]
                })
        return result_list

    # ------------------------------------------------------------------ #
    #  绘图（供外部调用，使用 result 列表）
    # ------------------------------------------------------------------ #
    def draw_img(self, img, result_box):
        for box_dict in result_box:
            clas_id = box_dict["cls_id"]
            label   = self.clas_names[clas_id]
            conf    = box_dict["conf"]
            x, y, w, h = box_dict["box"]
            color = self.color_list[clas_id]
            cv2.rectangle(img, (x, y), (x + w, y + h), color=color, thickness=3)
            cv2.putText(img, f"{label}_{conf:.2%}", (x, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        return img

    # ------------------------------------------------------------------ #
    #  NMS
    # ------------------------------------------------------------------ #
    def nms(self, dets, iou_thresh):
        boxes_area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        index = (-dets[:, -1]).argsort()
        keep = []

        def iou(box, boxes, box_area, boxes_area):
            xx1 = np.maximum(box[0], boxes[:, 0])
            yy1 = np.maximum(box[1], boxes[:, 1])
            xx2 = np.minimum(box[2], boxes[:, 2])
            yy2 = np.minimum(box[3], boxes[:, 3])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            ovr = inter / (box_area + boxes_area - inter)
            return ovr

        while index.size > 0:
            i = index[0]
            keep.append(i)
            idx = np.where(iou(dets[index[0]], dets[index[1:]],
                               boxes_area[index[0]], boxes_area[index[1:]]) <= iou_thresh)[0]
            index = index[idx + 1]
        return np.array(keep)

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
        min_wh, max_wh = 2, 4096
        output = [np.zeros((0, 6))] * prediction.shape[0]

        for xi, x in enumerate(prediction):
            if not x.shape[0]:
                continue
            box  = self.xywh2xyxy(x[:, :4])
            conf = np.expand_dims(x[:, 4:].max(1), 1)
            j    = np.expand_dims(x[:, 4:].argmax(1), 1)
            x    = np.concatenate((box, conf, j.astype(conf.dtype)), 1)[conf.reshape(-1) > conf_thres]

            if classes is not None:
                x = x[(x[:, 5:6] == np.array(classes)).any(1)]
            if not x.shape[0]:
                continue

            c = x[:, 5:6] * max_wh
            boxes, scores = x[:, :4] + c, x[:, 4:5]
            dets = np.concatenate((boxes, scores), 1)
            i = self.nms(dets, iou_thres)
            output[xi] = x[i]

        return output

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def scale_coords(self, boxes, pad, scale, image_shape):
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes[:, :4] /= scale

        def clip_coords(boxes, img_shape):
            boxes[:, 0] = boxes[:, 0].clip(0, img_shape[1])
            boxes[:, 1] = boxes[:, 1].clip(0, img_shape[0])
            boxes[:, 2] = boxes[:, 2].clip(0, img_shape[1])
            boxes[:, 3] = boxes[:, 3].clip(0, img_shape[0])

        clip_coords(boxes, image_shape)
        return boxes

    # ------------------------------------------------------------------ #
    #  颜色表
    # ------------------------------------------------------------------ #
    def SetColor(self, num_classes):
        arr_b = [0, 64, 255, 192, 128]
        arr_g = [255, 64, 128, 192, 0]
        arr_r = [128, 192, 0, 64, 255]
        color_list = []
        for a in arr_b:
            for b in arr_g:
                for c in arr_r:
                    color_list.append([a, b, c])
        return color_list

    # ------------------------------------------------------------------ #
    #  跨类 IoU 过滤
    # ------------------------------------------------------------------ #
    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        x1_i = max(x1_1, x1_2);  y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2);  y2_i = min(y2_1, y2_2)
        inter = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def process_detections(self, boxes, classes, scores, iou_threshold=0.5, priority_delete_class=None):
        """
        跨类 NMS，优先删除 priority_delete_class 类别的重叠框。
        参数与返回均为列表，顺序: boxes, classes(labels), scores
        """
        if not boxes:
            return [], [], []

        # 按置信度降序排列
        detections = sorted(zip(boxes, classes, scores), key=lambda x: x[2], reverse=True)
        suppressed = [False] * len(detections)
        keep = []

        for i in range(len(detections)):
            if suppressed[i]:
                continue
            box_i, class_i, score_i = detections[i]
            keep_current = True

            for j in range(len(detections)):
                if i == j or suppressed[j]:
                    continue
                box_j, class_j, score_j = detections[j]
                iou = self.calculate_iou(box_i, box_j)

                if iou > iou_threshold and class_i != class_j:
                    if class_i == priority_delete_class:
                        keep_current = False
                        suppressed[i] = True
                        break
                    elif class_j == priority_delete_class:
                        suppressed[j] = True
                    else:
                        if score_i < score_j:
                            keep_current = False
                            suppressed[i] = True
                            break
                        else:
                            suppressed[j] = True

            if keep_current and not suppressed[i]:
                keep.append(detections[i])

        if keep:
            out_boxes, out_classes, out_scores = zip(*keep)
            # 确保类型干净
            return (list(out_boxes),
                    [int(c) for c in out_classes],
                    [float(s) for s in out_scores])
        return [], [], []
