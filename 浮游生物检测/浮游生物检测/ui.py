import sys
import csv
import os
import datetime
import cv2
import torch
import numpy as np
from Alg import *
from collections import defaultdict

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QWidget, QFileDialog, QHBoxLayout, QSlider, QTableWidget, QTableWidgetItem,
    QLineEdit, QGroupBox, QGridLayout, QMessageBox, QHeaderView, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import time
import yaml
import matplotlib
matplotlib.use('Agg')                      # 非 GUI 后端，避免与 Qt 冲突
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties

# 每张图片对应的体积（单位：升）
IMG_VOLUME_L = 1.93321e-8


# ======================================================================
#  单张图片检测线程
# ======================================================================
class DetectionThread(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    error    = pyqtSignal(str)
    status   = pyqtSignal(str)

    def __init__(self, model, image_path, conf_thres, iou_thres):
        super().__init__()
        self.model      = model
        self.image_path = image_path
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            self.status.emit("正在读取图片...")
            self.progress.emit(10)

            if not os.path.exists(self.image_path):
                self.error.emit(f"图片文件不存在: {self.image_path}")
                return

            # 支持中文路径
            try:
                img_array = np.fromfile(self.image_path, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception:
                img = None
            if img is None:
                self.error.emit("无法读取图片，可能格式不支持或文件损坏")
                return

            self.status.emit("正在准备模型推理...")
            self.progress.emit(30)

            if self.model is None:
                self.error.emit("模型未正确加载，请重新加载模型")
                return

            self.status.emit("正在进行目标检测...")
            self.progress.emit(50)

            # Alg.detect 返回顺序: result, boxes, labels, scores
            #   boxes  : list of (x1,y1,x2,y2)
            #   labels : list of int  （类别索引）
            #   scores : list of float（置信度 0~1）
            result, boxes, labels, scores = self.model.detect(img.copy())

            # 只用 UI 循环绘制一遍，避免与 draw_img 重复造成重影
            result_img = img.copy()
            detections = []
            for box, label, score in zip(boxes, labels, scores):
                label = int(label)
                score = float(score)
                box   = [int(x) for x in box]

                cls_name   = self.model.clas_names[label]
                label_text = f'{cls_name}: {score:.2%}'
                color      = self.model.color_list[label]

                cv2.rectangle(result_img,
                              (box[0], box[1]), (box[2], box[3]),
                              color, thickness=4, lineType=cv2.LINE_AA)
                cv2.putText(result_img, label_text,
                            (box[0], max(box[1] - 10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                detections.append({
                    'id':         len(detections) + 1,
                    'class':      cls_name,
                    'confidence': score,
                    'bbox':       box
                })

            self.status.emit("正在处理检测结果...")
            self.progress.emit(70)

            self.status.emit("检测完成")
            self.progress.emit(100)

            self.finished.emit({
                'image':      result_img,
                'detections': detections,
                'filename':   os.path.basename(self.image_path),
                'width':      result_img.shape[1],
                'height':     result_img.shape[0],
                'filepath':   self.image_path,
            })

        except Exception as e:
            import traceback
            self.error.emit(f"检测过程错误: {str(e)}\n{traceback.format_exc()}")


# ======================================================================
#  批量检测线程
# ======================================================================
class BatchDetectionThread(QThread):
    # 信号: 进度(int), 文件名(str), 检测列表(list), 结果图(object/ndarray|None)
    progress = pyqtSignal(int, str, list, object)
    finished = pyqtSignal()
    error    = pyqtSignal(str)
    status   = pyqtSignal(str)

    def __init__(self, model, folder_path, conf_thres, iou_thres):
        super().__init__()
        self.model       = model
        self.folder_path = folder_path
        self.conf_thres  = conf_thres
        self.iou_thres   = iou_thres
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            self.status.emit("正在扫描图片文件...")

            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            image_files = [f for f in os.listdir(self.folder_path)
                           if os.path.splitext(f)[1].lower() in image_extensions]

            if not image_files:
                self.error.emit("文件夹中没有找到支持的图片文件")
                return

            total_files = len(image_files)
            self.status.emit(f"找到 {total_files} 张图片，开始检测...")

            for i, filename in enumerate(image_files):
                if not self._is_running:
                    break

                image_path = os.path.join(self.folder_path, filename)
                self.status.emit(f"正在检测: {filename}")

                try:
                    # 支持中文路径
                    try:
                        img_array = np.fromfile(image_path, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    except Exception:
                        img = None
                    if img is None:
                        raise Exception(f"无法读取图片: {filename}")

                    # Alg.detect 返回: result, boxes, labels, scores
                    result, boxes, labels, scores = self.model.detect(img.copy())

                    # 只用 UI 循环绘制一遍，避免与 draw_img 重复造成重影
                    draw_img   = img.copy()
                    detections = []
                    for box, label, score in zip(boxes, labels, scores):
                        label = int(label)
                        score = float(score)
                        box   = [int(x) for x in box]

                        cls_name   = self.model.clas_names[label]
                        color      = self.model.color_list[label]
                        label_text = f'{cls_name}: {score:.2%}'

                        cv2.rectangle(draw_img,
                                      (box[0], box[1]), (box[2], box[3]),
                                      color, thickness=4, lineType=cv2.LINE_AA)
                        cv2.putText(draw_img, label_text,
                                    (box[0], max(box[1] - 10, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                        detections.append({
                            'id':         len(detections) + 1,
                            'class':      cls_name,
                            'confidence': score,
                            'bbox':       box
                        })

                    result_img = np.ascontiguousarray(draw_img)
                    prog = int((i + 1) / total_files * 100)
                    self.progress.emit(prog, filename, detections, result_img)
                    time.sleep(0.05)

                except Exception as e:
                    prog = int((i + 1) / total_files * 100)
                    self.progress.emit(prog, filename, [], None)
                    continue

            self.status.emit("批量检测完成")
            self.finished.emit()

        except Exception as e:
            import traceback
            self.error.emit(f"批量检测错误: {str(e)}\n{traceback.format_exc()}")


# ======================================================================
#  主窗口
# ======================================================================
class DetectUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("浮游生物检测系统")
        self.resize(1400, 900)

        self.model              = None
        self.model_path         = ""
        self.conf_thres         = 0.25
        self.iou_thres          = 0.45
        self.current_image_path = None
        self.selected_folder    = None
        self.original_pixmap    = None
        self.detection_thread   = None
        self.batch_thread       = None
        self.batch_results      = {}
        self.single_result      = None          # 单张检测结果，含尺寸信息          # {filename: {'detections':[], 'image':ndarray}}
        self.class_counts       = defaultdict(int)
        self.total_detections   = 0

        self.init_ui()
        self.check_gpu_status()

    # ------------------------------------------------------------------ #
    #  界面初始化
    # ------------------------------------------------------------------ #
    def init_ui(self):
        root = QHBoxLayout()

        # ---------- 左侧：原图 / 结果图 ----------
        left = QVBoxLayout()
        self.label_input  = QLabel("原始图像")
        self.label_result = QLabel("检测结果")
        for lb in (self.label_input, self.label_result):
            lb.setAlignment(Qt.AlignCenter)
            lb.setStyleSheet("background-color:#f0f0f0; border:1px solid #c9c9c9;")
            lb.setMinimumSize(200, 200)
        left.addWidget(self.label_input,  1)
        left.addWidget(self.label_result, 1)

        # ---------- 右侧：控制区 ----------
        right = QVBoxLayout()

        # 模型设置
        gb_model = QGroupBox("模型设置")
        g = QGridLayout()
        self.edit_model = QLineEdit(); self.edit_model.setReadOnly(True)
        self.btn_pick_model  = QPushButton("选择模型文件…")
        self.btn_load_model  = QPushButton("加载模型")
        self.label_gpu_status = QLabel("GPU状态: 检测中...")
        self.label_gpu_status.setStyleSheet("color: blue;")
        self.btn_pick_model.clicked.connect(self.pick_model_file)
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_load_model.setEnabled(False)
        g.addWidget(QLabel("模型文件："), 0, 0)
        g.addWidget(self.edit_model,      0, 1, 1, 2)
        g.addWidget(self.btn_pick_model,  1, 0)
        g.addWidget(self.btn_load_model,  1, 1)
        g.addWidget(self.label_gpu_status, 2, 0, 1, 3)
        gb_model.setLayout(g)
        right.addWidget(gb_model)

        # 检测参数
        gb_param = QGroupBox("检测参数")
        gp = QGridLayout()
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(1, 100)
        self.slider_conf.setValue(int(self.conf_thres * 100))
        self.slider_conf.valueChanged.connect(self.update_conf_thres)
        self.slider_iou = QSlider(Qt.Horizontal)
        self.slider_iou.setRange(1, 100)
        self.slider_iou.setValue(int(self.iou_thres * 100))
        self.slider_iou.valueChanged.connect(self.update_iou_thres)
        self.label_conf_value = QLabel(f"{self.conf_thres:.2f}")
        self.label_iou_value  = QLabel(f"{self.iou_thres:.2f}")
        gp.addWidget(QLabel("置信度阈值："), 0, 0)
        gp.addWidget(self.slider_conf,      0, 1)
        gp.addWidget(self.label_conf_value, 0, 2)
        gp.addWidget(QLabel("IoU 阈值："),  1, 0)
        gp.addWidget(self.slider_iou,       1, 1)
        gp.addWidget(self.label_iou_value,  1, 2)
        gb_param.setLayout(gp)
        right.addWidget(gb_param)

        # 单张检测
        gb_single = QGroupBox("单张图片检测")
        sl = QVBoxLayout()
        sb = QHBoxLayout()
        self.btn_select_image = QPushButton("选择图片")
        self.btn_start_detect = QPushButton("开始检测")
        self.btn_select_image.clicked.connect(self.select_image)
        self.btn_start_detect.clicked.connect(self.start_detection)
        sb.addWidget(self.btn_select_image)
        sb.addWidget(self.btn_start_detect)
        sl.addLayout(sb)
        self.label_status = QLabel("就绪")
        self.label_status.setAlignment(Qt.AlignCenter)
        sl.addWidget(self.label_status)
        self.single_progress_bar = QProgressBar()
        self.single_progress_bar.setVisible(False)
        sl.addWidget(self.single_progress_bar)
        gb_single.setLayout(sl)
        right.addWidget(gb_single)

        # 批量检测
        gb_batch = QGroupBox("批量文件夹检测")
        bl = QVBoxLayout()
        bb = QHBoxLayout()
        self.btn_select_folder = QPushButton("选择文件夹")
        self.btn_start_batch   = QPushButton("开始批量检测")
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.btn_start_batch.clicked.connect(self.start_batch_detection)
        bb.addWidget(self.btn_select_folder)
        bb.addWidget(self.btn_start_batch)
        bl.addLayout(bb)
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setVisible(False)
        bl.addWidget(self.batch_progress_bar)
        gb_batch.setLayout(bl)
        right.addWidget(gb_batch)

        # 控制按钮
        gb_ctrl = QGroupBox("控制")
        cl = QHBoxLayout()
        self.btn_stop          = QPushButton("停止检测")
        self.btn_export        = QPushButton("导出检测结果")
        self.btn_clear_summary = QPushButton("清空汇总")
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_clear_summary.clicked.connect(self.clear_summary)
        cl.addWidget(self.btn_stop)
        cl.addWidget(self.btn_export)
        cl.addWidget(self.btn_clear_summary)
        gb_ctrl.setLayout(cl)
        right.addWidget(gb_ctrl)

        # 批量汇总表（每图一行）
        gb_summary = QGroupBox("批量检测汇总（每张图片）")
        sum_layout = QVBoxLayout()
        sum_info   = QHBoxLayout()
        self.label_total_images     = QLabel("图片数量: 0")
        self.label_total_detections = QLabel("检测总数: 0")
        sum_info.addWidget(self.label_total_images)
        sum_info.addWidget(self.label_total_detections)
        sum_layout.addLayout(sum_info)
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(3)
        self.summary_table.setHorizontalHeaderLabels(["文件名", "检测数量", "各类别统计"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.setMaximumHeight(180)
        self.summary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.summary_table.cellClicked.connect(self.on_summary_row_clicked)
        sum_layout.addWidget(self.summary_table)
        gb_summary.setLayout(sum_layout)
        right.addWidget(gb_summary)

        # 明细表（点击汇总行显示）
        gb_detail = QGroupBox("检测明细（点击上方图片行查看详情）")
        det_layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["序号", "类别", "置信度", "xmin", "ymin", "xmax", "ymax"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        det_layout.addWidget(self.table)
        gb_detail.setLayout(det_layout)
        right.addWidget(gb_detail, 1)

        # 组装
        root.addLayout(left,  7)
        root.addLayout(right, 3)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        # 初始按钮状态
        self.btn_select_image.setEnabled(False)
        self.btn_start_detect.setEnabled(False)
        self.btn_select_folder.setEnabled(False)
        self.btn_start_batch.setEnabled(False)
        self.btn_stop.setEnabled(False)

    # ------------------------------------------------------------------ #
    #  GPU 状态
    # ------------------------------------------------------------------ #
    def check_gpu_status(self):
        try:
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                cnt  = torch.cuda.device_count()
                self.label_gpu_status.setText(f"GPU状态: 可用 ({name}, 设备数:{cnt})")
                self.label_gpu_status.setStyleSheet("color: green;")
            else:
                self.label_gpu_status.setText("GPU状态: 不可用（使用CPU）")
                self.label_gpu_status.setStyleSheet("color: red;")
        except Exception:
            self.label_gpu_status.setText("GPU状态: 检测失败")
            self.label_gpu_status.setStyleSheet("color: orange;")

    # ------------------------------------------------------------------ #
    #  模型加载
    # ------------------------------------------------------------------ #
    def pick_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "",
                                              "Model Files (*.pt *.onnx);;All Files (*.*)")
        if not path:
            return
        self.model_path = path
        self.edit_model.setText(path)
        self.btn_load_model.setEnabled(True)

    def load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            QMessageBox.warning(self, "错误", "模型文件不存在！")
            return
        try:
            self.label_status.setText("正在加载模型...")
            QApplication.processEvents()

            # 自动从模型同目录寻找 alg_labels.yaml
            model_dir  = os.path.dirname(os.path.abspath(self.model_path))
            label_path = os.path.join(model_dir, "alg_labels.yaml")
            if not os.path.exists(label_path):
                label_path = os.path.join(os.getcwd(), "alg_labels.yaml")
            if not os.path.exists(label_path):
                raise Exception(f"找不到 alg_labels.yaml，请放到模型同目录：{model_dir}")

            with open(label_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f.read())
            names_field = yaml_data["names"]
            if isinstance(names_field, dict):
                clas_names = list(names_field.values())
            elif isinstance(names_field, list):
                clas_names = names_field
            else:
                raise Exception(f"yaml 中 names 字段格式不支持: {type(names_field)}")

            self.model = yolo_onnx(self.model_path, clas_names, conf_thres=0.30)
            if self.model is None:
                raise Exception("模型加载返回 None")

            QMessageBox.information(self, "成功", "模型加载成功！")
            self.btn_select_image.setEnabled(True)
            self.btn_select_folder.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.label_status.setText("模型加载成功，就绪")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            self.label_status.setText("模型加载失败")

    # ------------------------------------------------------------------ #
    #  参数更新
    # ------------------------------------------------------------------ #
    def update_conf_thres(self, value):
        self.conf_thres = value / 100
        self.label_conf_value.setText(f"{self.conf_thres:.2f}")

    def update_iou_thres(self, value):
        self.iou_thres = value / 100
        self.label_iou_value.setText(f"{self.iou_thres:.2f}")

    # ------------------------------------------------------------------ #
    #  单张检测
    # ------------------------------------------------------------------ #
    def select_image(self):
        if self.model is None:
            QMessageBox.warning(self, "错误", "请先加载模型！")
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "",
                                              "Images (*.jpg *.jpeg *.png *.bmp *.tif)")
        if not path:
            return
        self.current_image_path = path
        self.original_pixmap    = QPixmap(path)
        if self.original_pixmap.isNull():
            self.label_input.setText("无法加载图片")
            return
        self._show_pixmap(self.label_input, self.original_pixmap)
        self.btn_start_detect.setEnabled(True)
        self.label_status.setText("图片已选择，点击开始检测")
        self.label_result.setText("点击'开始检测'进行检测")
        self.table.setRowCount(0)

    def start_detection(self):
        if self.model is None or self.current_image_path is None:
            QMessageBox.warning(self, "错误", "请先选择模型和图片！")
            return
        self.btn_start_detect.setEnabled(False)
        self.btn_select_image.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.label_result.setText("检测中...")
        self.label_status.setText("检测中...")
        self.single_progress_bar.setVisible(True)
        self.single_progress_bar.setValue(0)

        self.detection_thread = DetectionThread(
            self.model, self.current_image_path, self.conf_thres, self.iou_thres)
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.progress.connect(self.single_progress_bar.setValue)
        self.detection_thread.error.connect(self.on_detection_error)
        self.detection_thread.status.connect(self.label_status.setText)
        self.detection_thread.start()

    def on_detection_finished(self, result):
        self.single_progress_bar.setVisible(False)
        # 显示结果图
        img_c = np.ascontiguousarray(result['image'])
        h, w  = img_c.shape[:2]
        q_img = QImage(img_c.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        self._show_pixmap(self.label_result, QPixmap.fromImage(q_img))
        # 更新明细表
        self.update_detail_table(result['detections'])
        # 存储单张结果（含尺寸），供导出 XML 使用
        self.single_result = result
        self.btn_start_detect.setEnabled(True)
        self.btn_select_image.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.label_status.setText(f"检测完成，发现 {len(result['detections'])} 个目标")

    def on_detection_error(self, error_msg):
        self.single_progress_bar.setVisible(False)
        QMessageBox.critical(self, "检测错误", error_msg)
        self.label_result.setText("检测失败")
        self.label_status.setText("检测失败")
        self.btn_start_detect.setEnabled(True)
        self.btn_select_image.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ------------------------------------------------------------------ #
    #  批量检测
    # ------------------------------------------------------------------ #
    def select_folder(self):
        if self.model is None:
            QMessageBox.warning(self, "错误", "请先加载模型！")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择包含图片的文件夹")
        if not folder:
            return
        self.selected_folder = folder
        exts  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        files = [f for f in os.listdir(folder)
                 if os.path.splitext(f)[1].lower() in exts]
        if not files:
            QMessageBox.warning(self, "提示", "该文件夹中没有图片！")
            return
        self.btn_start_batch.setEnabled(True)
        self.table.setRowCount(0)
        self.label_status.setText(f"已选择文件夹，包含 {len(files)} 张图片")
        QMessageBox.information(self, "选择完成",
                                f"已选择文件夹，包含 {len(files)} 张图片。\n点击'开始批量检测'开始。")

    def start_batch_detection(self):
        if not self.selected_folder:
            QMessageBox.warning(self, "错误", "请先选择文件夹！")
            return
        self.batch_results = {}
        self.class_counts.clear()
        self.total_detections = 0
        self.update_summary_table()
        self.batch_progress_bar.setVisible(True)
        self.batch_progress_bar.setValue(0)
        self.btn_start_batch.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.label_status.setText("批量检测中...")

        self.batch_thread = BatchDetectionThread(
            self.model, self.selected_folder, self.conf_thres, self.iou_thres)
        self.batch_thread.progress.connect(self.on_batch_progress)
        self.batch_thread.finished.connect(self.on_batch_finished)
        self.batch_thread.error.connect(self.on_batch_error)
        self.batch_thread.status.connect(self.label_status.setText)
        self.batch_thread.start()

    def on_batch_progress(self, progress, filename, detections, result_img):
        self.batch_progress_bar.setValue(progress)
        # 同时存图片尺寸，供生成 XML 使用
        img_h, img_w = (result_img.shape[:2] if result_img is not None else (0, 0))
        self.batch_results[filename] = {
            'detections': detections,
            'image':      result_img,
            'width':      img_w,
            'height':     img_h,
        }
        self.update_summary_table()

    def on_summary_row_clicked(self, row, col):
        """点击汇总行：左侧显示原图+结果图，明细表显示检测框"""
        item = self.summary_table.item(row, 0)
        if item is None:
            return
        filename   = item.text()
        data       = self.batch_results.get(filename, {})
        detections = data.get('detections', [])
        result_img = data.get('image', None)

        # 明细表
        self.update_detail_table(detections)

        # 左下：结果图
        if result_img is not None:
            img_c = np.ascontiguousarray(result_img)
            h, w  = img_c.shape[:2]
            q_img = QImage(img_c.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
            self._show_pixmap(self.label_result, QPixmap.fromImage(q_img))
        else:
            self.label_result.setText("该图片检测失败或无结果")

        # 左上：原始图
        if self.selected_folder:
            orig_path = os.path.join(self.selected_folder, filename)
            if os.path.exists(orig_path):
                pix = QPixmap(orig_path)
                if not pix.isNull():
                    self._show_pixmap(self.label_input, pix)

        self.label_status.setText(f"查看: {filename}，共 {len(detections)} 个目标")

    def on_batch_finished(self):
        self.batch_progress_bar.setVisible(False)
        self.btn_start_batch.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.label_status.setText("批量检测完成")
        QMessageBox.information(self, "完成", "批量检测完成！")

    def on_batch_error(self, error_msg):
        self.batch_progress_bar.setVisible(False)
        self.btn_start_batch.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.label_status.setText("批量检测错误")
        QMessageBox.critical(self, "错误", error_msg)

    # ------------------------------------------------------------------ #
    #  停止检测
    # ------------------------------------------------------------------ #
    def stop_detection(self):
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.detection_thread.wait(1000)
            self.btn_start_detect.setEnabled(True)
            self.btn_select_image.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.single_progress_bar.setVisible(False)
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.stop()
            self.batch_thread.wait(1000)
            self.batch_progress_bar.setVisible(False)
            self.btn_start_batch.setEnabled(True)
            self.btn_stop.setEnabled(False)
        self.label_status.setText("检测已停止")
        QMessageBox.information(self, "提示", "检测已停止")

    # ------------------------------------------------------------------ #
    #  生成 Pascal VOC XML 标注
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_xml(filename, filepath, img_w, img_h, detections):
        """生成 Pascal VOC 格式的 XML 字符串"""
        lines = []
        lines.append('<annotation>')
        lines.append(f'\t<folder>{os.path.dirname(filepath)}</folder>')
        lines.append(f'\t<filename>{filename}</filename>')
        lines.append(f'\t<path>{filepath}</path>')
        lines.append('\t<source><database>Unknown</database></source>')
        lines.append('\t<size>')
        lines.append(f'\t\t<width>{img_w}</width>')
        lines.append(f'\t\t<height>{img_h}</height>')
        lines.append('\t\t<depth>3</depth>')
        lines.append('\t</size>')
        lines.append('\t<segmented>0</segmented>')
        for det in detections:
            b = det['bbox']   # [x1, y1, x2, y2]
            lines.append('\t<object>')
            lines.append(f'\t\t<name>{det["class"]}</name>')
            lines.append('\t\t<pose>Unspecified</pose>')
            lines.append('\t\t<truncated>0</truncated>')
            lines.append('\t\t<difficult>0</difficult>')
            lines.append(f'\t\t<confidence>{det["confidence"]:.4f}</confidence>')
            lines.append('\t\t<bndbox>')
            lines.append(f'\t\t\t<xmin>{b[0]}</xmin>')
            lines.append(f'\t\t\t<ymin>{b[1]}</ymin>')
            lines.append(f'\t\t\t<xmax>{b[2]}</xmax>')
            lines.append(f'\t\t\t<ymax>{b[3]}</ymax>')
            lines.append('\t\t</bndbox>')
            lines.append('\t</object>')
        lines.append('</annotation>')
        return '\n'.join(lines)

    # ------------------------------------------------------------------ #
    #  密度统计核心计算
    # ------------------------------------------------------------------ #
    @staticmethod
    def _calc_density(batch_results, img_volume_l):
        """
        返回:
            per_img_density : { filename: { cls: density(Cells/L) } }
            cls_avg_density : { cls: 所有图片平均密度 }
            total_avg_density : float  所有类合计平均密度
        """
        # 收集每张图每类的个数
        per_img_count = {}     # { filename: {cls: count} }
        all_cls = set()
        for filename, data in batch_results.items():
            dets = data.get('detections', []) if isinstance(data, dict) else data
            cnt  = defaultdict(int)
            for d in dets:
                cnt[d['class']] += 1
                all_cls.add(d['class'])
            per_img_count[filename] = cnt

        n_imgs = len(per_img_count)
        if n_imgs == 0 or img_volume_l == 0:
            return {}, {}, 0.0

        # 每张图每类密度
        per_img_density = {}
        for fn, cnt in per_img_count.items():
            per_img_density[fn] = {cls: cnt.get(cls, 0) / img_volume_l for cls in all_cls}

        # 各类平均密度（对所有图片取均值）
        cls_avg_density = {}
        for cls in all_cls:
            cls_avg_density[cls] = sum(
                per_img_density[fn][cls] for fn in per_img_density
            ) / n_imgs

        total_avg_density = sum(cls_avg_density.values())
        return per_img_density, cls_avg_density, total_avg_density

    # ------------------------------------------------------------------ #
    #  生成图表（饼图 + 柱状图），保存为 PNG
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_charts(cls_avg_density, total_avg_density, save_dir, ts):
        """
        生成两张图:
          藻密度占比_<ts>.png   各类藻占比饼图
          藻密度柱状图_<ts>.png 总藻密度 & 各类藻密度柱状图
        两图颜色与排序完全一致（均按密度从高到低）
        返回 (pie_path, bar_path)
        """
        # ── 中文字体 ──────────────────────────────────────────────────────
        try:
            fp = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
        except Exception:
            try:
                fp = FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
            except Exception:
                fp = FontProperties()

        # ── 排序（密度从高到低），饼图与柱状图共用同一顺序 ──────────────
        cls_list  = sorted(cls_avg_density.keys(),
                           key=lambda c: cls_avg_density[c], reverse=True)
        densities = [cls_avg_density[c] for c in cls_list]

        # ── 期刊配色（Nature/Science 色盲友好色板） ──────────────────────
        #   JOURNAL_COLORS[0]  = 总藻密度专用（钢蓝）
        #   JOURNAL_COLORS[1:] = 各类藻，按 cls_list 顺序依次分配
        JOURNAL_COLORS = [
            "#4878CF",  # 钢蓝   — 总藻密度专用
            "#D65F5F",  # 砖红
            "#6ACC65",  # 草绿
            "#B47CC7",  # 淡紫
            "#C4AD66",  # 卡其
            "#77BEDB",  # 天蓝
            "#E58C8A",  # 玫瑰
            "#56B4E9",  # 明蓝
            "#009E73",  # 翠绿
            "#E69F00",  # 琥珀
            "#CC79A7",  # 洋红
            "#0072B2",  # 深蓝
        ]
        # ★ 共享颜色映射字典：cls_list[i] → JOURNAL_COLORS[i+1]
        #   饼图和柱状图均从此字典取色，确保完全一致
        color_map = {
            cls: JOURNAL_COLORS[(i + 1) % len(JOURNAL_COLORS)]
            for i, cls in enumerate(cls_list)
        }
        cls_colors = [color_map[c] for c in cls_list]  # 按密度降序的颜色列表

        # ── 全局样式（期刊规范：无上/右边框，淡灰横向网格）────────────────
        plt.rcParams.update({
            'axes.spines.top':   False,
            'axes.spines.right': False,
            'axes.grid':         True,
            'axes.grid.axis':    'y',
            'grid.color':        '#E0E0E0',
            'grid.linewidth':    0.6,
            'figure.facecolor':  'white',
            'axes.facecolor':    'white',
        })

        # ════════════════════════════════════════════════════════════════
        #  饼图
        # ════════════════════════════════════════════════════════════════
        pie_path = os.path.join(save_dir, f"藻密度占比_{ts}.png")
        fig1, ax1 = plt.subplots(figsize=(8, 7))

        # 饼图数据与颜色均使用与柱状图相同的 cls_list 顺序
        wedges, _, autotexts = ax1.pie(
            densities,
            labels=None,                          # 标签放图例，不放扇区
            colors=cls_colors,                    # ★ 使用共享颜色映射
            autopct='%1.1f%%',
            startangle=90,                        # 从12点钟方向顺时针绘制
            counterclock=False,                   # 顺时针 → 最大扇区在最显眼位置
            pctdistance=0.78,
            wedgeprops=dict(linewidth=1.0, edgecolor='white'),
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_color('white')
            at.set_fontweight('bold')

        ax1.set_title("各类藻密度占比", fontproperties=fp, fontsize=15,
                      pad=18, fontweight='bold')

        # 图例：顺序与 cls_list 完全一致（密度降序），颜色与柱状图一一对应
        legend_labels = [f"类{c}:  {cls_avg_density[c]:.2e} Cells/L"
                         for c in cls_list]
        ax1.legend(wedges, legend_labels,
                   loc="lower center", bbox_to_anchor=(0.5, -0.22),
                   ncol=min(3, len(cls_list)), prop=fp, fontsize=10,
                   frameon=False)

        plt.tight_layout()
        fig1.savefig(pie_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # ════════════════════════════════════════════════════════════════
        #  柱状图
        # ════════════════════════════════════════════════════════════════
        bar_path   = os.path.join(save_dir, f"藻密度柱状图_{ts}.png")
        bar_labels = ["总藻密度"] + [f"类{c}" for c in cls_list]
        bar_values = [total_avg_density] + densities
        bar_colors = [JOURNAL_COLORS[0]] + cls_colors   # ★ 总藻密度钢蓝 + 各类共享颜色

        fig2, ax2 = plt.subplots(figsize=(max(7, len(bar_labels) * 1.5), 6))
        bars = ax2.bar(range(len(bar_labels)), bar_values,
                       color=bar_colors, width=0.6,
                       edgecolor='white', linewidth=0.8,
                       zorder=3)

        ax2.set_xticks(range(len(bar_labels)))
        ax2.set_xticklabels(bar_labels, fontproperties=fp, fontsize=12,
                            rotation=30, ha='right')
        ax2.set_ylabel("密度 (Cells/L)", fontproperties=fp, fontsize=13)
        ax2.set_title("总藻密度及各类藻密度", fontproperties=fp, fontsize=15,
                      fontweight='bold', pad=12)
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.2e}"))
        ax2.tick_params(axis='y', labelsize=10)
        ax2.tick_params(axis='x', length=0)
        ax2.set_axisbelow(True)

        # 柱顶数值标注
        for bar, val in zip(bars, bar_values):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + total_avg_density * 0.008,
                     f"{val:.2e}",
                     ha='center', va='bottom', fontsize=9,
                     color='#333333', fontweight='bold')

        # 总藻密度参考虚线
        ax2.axhline(total_avg_density, color=JOURNAL_COLORS[0],
                    linewidth=1.0, linestyle='--', alpha=0.45, zorder=2)

        # 柱状图图例（与饼图颜色完全一致）
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=color_map[c], edgecolor='white', label=f"类{c}")
            for c in cls_list
        ]
        ax2.legend(handles=legend_handles,
                   loc="upper right", prop=fp, fontsize=9,
                   frameon=True, framealpha=0.8, edgecolor='#CCCCCC',
                   ncol=min(3, len(cls_list)))

        plt.tight_layout()
        fig2.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)

        # 恢复默认样式
        plt.rcParams.update(plt.rcParamsDefault)

        return pie_path, bar_path

    # ------------------------------------------------------------------ #
    #  导出 CSV（+ 可选 XML）
    # ------------------------------------------------------------------ #
    def export_csv(self):
        has_batch  = len(self.batch_results) > 0
        has_single = self.single_result is not None or self.table.rowCount() > 0
        if not has_batch and not has_single:
            QMessageBox.warning(self, "提示", "没有检测结果可导出！")
            return

        # ---- 询问是否同时导出 XML 标注文件 ----
        reply_xml = QMessageBox.question(
            self, "导出选项 (1/2)",
            "是否同时导出每张图片对应的 XML 标注文件（Pascal VOC 格式）？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        export_xml = (reply_xml == QMessageBox.Yes)

        # ---- 询问是否导出密度统计及图表（仅批量模式有意义）----
        export_density = False
        if has_batch:
            reply_den = QMessageBox.question(
                self, "导出选项 (2/2)",
                "是否同时导出藻密度统计 CSV 及饼图、柱状图？\n"
                f"（体积基准：{IMG_VOLUME_L:.5e} L/张，单位 Cells/L）",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            export_density = (reply_den == QMessageBox.Yes)

        # ---- 选择主 CSV 保存路径 ----
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_path, _ = QFileDialog.getSaveFileName(
            self, "导出检测明细（CSV）", f"检测结果_{ts}.csv", "CSV Files (*.csv)")
        if not csv_path:
            return
        save_dir = os.path.dirname(csv_path)

        # ---- XML 目录 ----
        xml_dir = None
        if export_xml:
            xml_dir = QFileDialog.getExistingDirectory(self, "选择 XML 文件保存目录")
            if not xml_dir:
                export_xml = False

        try:
            xml_count = 0

            # ================================================================
            #  写明细 CSV
            # ================================================================
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["序号", "文件名", "类别", "置信度", "xmin", "ymin", "xmax", "ymax"])

                if has_batch:
                    seq = 1
                    for filename, data in self.batch_results.items():
                        dets  = data.get('detections', []) if isinstance(data, dict) else data
                        img_w = data.get('width',  0) if isinstance(data, dict) else 0
                        img_h = data.get('height', 0) if isinstance(data, dict) else 0
                        for det in dets:
                            b = det['bbox']
                            writer.writerow([seq, filename, det['class'],
                                             f"{det['confidence']:.4f}",
                                             b[0], b[1], b[2], b[3]])
                            seq += 1
                        if export_xml and xml_dir:
                            img_path = os.path.join(
                                self.selected_folder if self.selected_folder else "", filename)
                            xml_str  = self._make_xml(filename, img_path, img_w, img_h, dets)
                            xml_name = os.path.splitext(filename)[0] + ".xml"
                            with open(os.path.join(xml_dir, xml_name), "w", encoding="utf-8") as xf:
                                xf.write(xml_str)
                            xml_count += 1
                else:
                    sr    = self.single_result
                    dets  = sr['detections'] if sr else []
                    fn    = sr['filename']   if sr else "unknown"
                    img_w = sr.get('width',  0) if sr else 0
                    img_h = sr.get('height', 0) if sr else 0
                    fp_sr = sr.get('filepath', fn) if sr else fn
                    for i, det in enumerate(dets):
                        b = det['bbox']
                        writer.writerow([i + 1, fn, det['class'],
                                         f"{det['confidence']:.4f}",
                                         b[0], b[1], b[2], b[3]])
                    if export_xml and xml_dir and sr:
                        xml_str  = self._make_xml(fn, fp_sr, img_w, img_h, dets)
                        xml_name = os.path.splitext(fn)[0] + ".xml"
                        with open(os.path.join(xml_dir, xml_name), "w", encoding="utf-8") as xf:
                            xf.write(xml_str)
                        xml_count += 1

            # ================================================================
            #  密度统计 CSV + 图表
            # ================================================================
            pie_path = bar_path = density_csv_path = None
            if export_density and has_batch:
                per_img_density, cls_avg_density, total_avg_density = \
                    self._calc_density(self.batch_results, IMG_VOLUME_L)

                # -- 密度统计 CSV --
                density_csv_path = os.path.join(save_dir, f"藻密度统计_{ts}.csv")
                with open(density_csv_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    all_cls = sorted(cls_avg_density.keys())

                    # 逐图明细
                    writer.writerow(["=== 每张图片各类藻密度 (Cells/L) ==="])
                    writer.writerow(["文件名"] + all_cls + ["合计密度"])
                    for fn, cls_dens in per_img_density.items():
                        row_total = sum(cls_dens.values())
                        writer.writerow(
                            [fn]
                            + [f"{cls_dens.get(c, 0):.2e}" for c in all_cls]
                            + [f"{row_total:.2e}"]
                        )

                    writer.writerow([])  # 空行
                    # 汇总行
                    writer.writerow(["=== 各类藻平均密度汇总 (Cells/L) ==="])
                    writer.writerow(["类别", "平均密度 (Cells/L)", "占总密度比例"])
                    for cls in all_cls:
                        ratio = cls_avg_density[cls] / total_avg_density if total_avg_density else 0
                        writer.writerow([cls,
                                         f"{cls_avg_density[cls]:.2e}",
                                         f"{ratio:.2%}"])
                    writer.writerow(["总藻平均密度", f"{total_avg_density:.2e}", "100%"])
                    writer.writerow([])
                    writer.writerow(["体积基准（L/张）", f"{IMG_VOLUME_L:.5e}"])
                    writer.writerow(["图片张数", len(self.batch_results)])

                # -- 生成图表 --
                if cls_avg_density:
                    pie_path, bar_path = self._make_charts(
                        cls_avg_density, total_avg_density, save_dir, ts)

            # ================================================================
            #  完成提示
            # ================================================================
            msg = f"✅ 检测明细 CSV:\n{csv_path}"
            if export_xml and xml_count > 0:
                msg += f"\n\n✅ XML 标注文件（{xml_count} 个）:\n{xml_dir}"
            if density_csv_path:
                msg += f"\n\n✅ 藻密度统计 CSV:\n{density_csv_path}"
            if pie_path:
                msg += f"\n\n✅ 饼图: {os.path.basename(pie_path)}"
            if bar_path:
                msg += f"\n✅ 柱状图: {os.path.basename(bar_path)}"
            if export_density and has_batch:
                msg += f"\n\n📊 总藻平均密度: {total_avg_density:.2e} Cells/L"
            QMessageBox.information(self, "导出成功", msg)

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "导出失败", f"{str(e)}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------ #
    #  汇总表（批量）
    # ------------------------------------------------------------------ #
    def update_summary_table(self):
        """每张图片一行：文件名 | 检测数量 | 各类别统计"""
        self.summary_table.setRowCount(0)
        total_det = 0
        for idx, (filename, data) in enumerate(self.batch_results.items()):
            dets = data.get('detections', []) if isinstance(data, dict) else data
            self.summary_table.insertRow(idx)
            self.summary_table.setItem(idx, 0, QTableWidgetItem(filename))
            self.summary_table.setItem(idx, 1, QTableWidgetItem(str(len(dets))))
            counter = defaultdict(int)
            for d in dets:
                counter[d['class']] += 1
            cls_str = ", ".join(f"{k}:{v}" for k, v in sorted(counter.items(), key=lambda x: x[1], reverse=True))
            self.summary_table.setItem(idx, 2, QTableWidgetItem(cls_str or "无目标"))
            total_det += len(dets)
        self.label_total_images.setText(f"图片数量: {len(self.batch_results)}")
        self.label_total_detections.setText(f"检测总数: {total_det}")

    def clear_summary(self):
        self.class_counts.clear()
        self.total_detections = 0
        self.batch_results.clear()
        self.summary_table.setRowCount(0)
        self.table.setRowCount(0)
        self.label_result.setText("检测结果")
        self.label_input.setText("原始图像")
        self.label_total_images.setText("图片数量: 0")
        self.label_total_detections.setText("检测总数: 0")
        QMessageBox.information(self, "提示", "汇总数据已清空！")

    # ------------------------------------------------------------------ #
    #  明细表
    # ------------------------------------------------------------------ #
    def update_detail_table(self, detections):
        """
        明细表列: 序号 | 类别 | 置信度(%) | xmin | ymin | xmax | ymax
        置信度以百分比字符串显示，如 "19.00%"
        """
        self.table.setRowCount(0)
        for i, det in enumerate(detections):
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(str(det['id'])))
            self.table.setItem(i, 1, QTableWidgetItem(str(det['class'])))
            # confidence 是 0~1 浮点，显示为百分比
            self.table.setItem(i, 2, QTableWidgetItem(f"{det['confidence']:.2%}"))
            for j, coord in enumerate(det['bbox']):
                if j < 4:
                    self.table.setItem(i, 3 + j, QTableWidgetItem(str(coord)))

    def update_table(self, filename, detections):
        """单张检测完成后调用"""
        self.update_detail_table(detections)

    # ------------------------------------------------------------------ #
    #  工具函数
    # ------------------------------------------------------------------ #
    def _show_pixmap(self, label: QLabel, pixmap: QPixmap):
        scaled = pixmap.scaled(label.width(), label.height(),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.original_pixmap and not self.original_pixmap.isNull():
            self._show_pixmap(self.label_input, self.original_pixmap)

    def closeEvent(self, event):
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.detection_thread.wait(2000)
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.stop()
            self.batch_thread.wait(2000)
        event.accept()


# ======================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("浮游生物检测系统")
    app.setApplicationVersion("1.0")
    w = DetectUI()
    w.show()
    sys.exit(app.exec_())
