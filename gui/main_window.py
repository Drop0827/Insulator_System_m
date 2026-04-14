import sys
import os
import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QListWidget, QGroupBox, QSplitter,
    QTableWidget, QTableWidgetItem, QDialog, QMessageBox, QScrollArea, QFrame,
    QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

class ImagePreviewLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.orig_pixmap = QPixmap()
        self.setAlignment(Qt.AlignCenter)
        # 核心修复：设置 SizePolicy 为 Ignored ！！！
        # 这确保了 QLabel 不会根据其图片内容去撑开布局。
        # 它是导致“无限放大”反馈环路的根本原因。
        from PyQt5.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(10, 10) # 允许自由缩小
        self._is_scaling = False
        
    def setPixmapWithResize(self, pixmap):
        self.orig_pixmap = pixmap
        if not self.orig_pixmap.isNull():
            super().setPixmap(self.orig_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._is_scaling and hasattr(self, 'orig_pixmap') and not self.orig_pixmap.isNull():
            self._is_scaling = True
            try:
                # 只在当前分配到的空间内缩放，由于 Policy 是 Ignored，它不会反向撑大父容器
                super().setPixmap(self.orig_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            finally:
                self._is_scaling = False

class ClickableLabel(ImagePreviewLabel):
    clicked = pyqtSignal(QLabel)
    def __init__(self, text=""):
        super().__init__(text)
        
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self)

from inference.detector_thread import YoloInferenceThread
from db.models import get_session, InspectionRecord, DetectionResult

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("输电线路绝缘子故障智能诊断与性能推演平台 — 实验版本")
        self.resize(1340, 950)

        p_best = r"C:\Develop\BS\Insulator_System_m\runs\train\yolo11_WTConv_exp\weights\yolo11_WTConv_exp.pt"
        self.compare_model_paths = [p_best] if os.path.exists(p_best) else []
        self.inference_thread = None

        self._init_ui()
        self._refresh_compare_ui()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        top_bar = QHBoxLayout()
        self.lbl_model_info = QLabel(f"对比队列: {len(self.compare_model_paths)}")
        self.lbl_model_info.setStyleSheet("color: #0078D4; font-weight: bold; margin-right: 15px;")
        
        self.btn_load_model = QPushButton("📂 载入权重模型")
        self.btn_upload_img = QPushButton("🖼️ 图片分析")
        self.btn_upload_vid = QPushButton("🎞️ 视频监测")
        self.btn_stop = QPushButton("🛑 终止")
        self.btn_history = QPushButton("📈 检修历史数据")
        
        btn_qss = "padding: 8px 15px; border-radius: 4px; font-weight: bold;"
        self.btn_load_model.setStyleSheet("background-color: #0078D4; color: white;" + btn_qss)
        self.btn_history.setStyleSheet("background-color: #28a745; color: white;" + btn_qss)
        
        top_bar.addWidget(self.lbl_model_info)
        top_bar.addWidget(self.btn_load_model)
        top_bar.addWidget(self.btn_upload_img)
        top_bar.addWidget(self.btn_upload_vid)
        top_bar.addWidget(self.btn_stop)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_history)
        main_layout.addLayout(top_bar)
        
        main_splitter = QSplitter(Qt.Vertical)

        # ====== 上半部分视窗 ======
        top_splitter = QSplitter(Qt.Horizontal)
        
        left_group = QGroupBox("诊断视图")
        self.lbl_display = ImagePreviewLabel("等待载入...")
        self.lbl_display.setMinimumSize(700, 420)
        self.lbl_display.setStyleSheet("background-color: #f7f7f9; border: 1px solid #ccc;")
        l_layout = QVBoxLayout(left_group); l_layout.addWidget(self.lbl_display)
        
        right_group = QGroupBox("检测结果汇总")
        r_layout = QVBoxLayout(right_group)
        
        self.list_logs = QListWidget()
        self.list_logs.setStyleSheet("border: 1px solid #ddd; font-family: Consolas;")
        
        lbl_log = QLabel("📝 算法检测日志")
        lbl_log.setStyleSheet("font-weight: bold; margin-top: 5px;")
        r_layout.addWidget(lbl_log)
        r_layout.addWidget(self.list_logs, stretch=1)
        
        lbl_tab_layout = QHBoxLayout()
        lbl_tab = QLabel("📊 真值锚框(GT)对比视图")
        lbl_tab.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.btn_upload_gt = QPushButton("📂 导入真值(GT)")
        self.btn_upload_gt.setStyleSheet("background-color: #f0f0f0; border-radius: 3px; padding: 4px; border: 1px solid #ccc; margin-top: 10px;")
        lbl_tab_layout.addWidget(lbl_tab)
        lbl_tab_layout.addStretch()
        lbl_tab_layout.addWidget(self.btn_upload_gt)
        r_layout.addLayout(lbl_tab_layout)
        
        self.lbl_gt_display = ImagePreviewLabel("等待用户导入GT真值坐标文件 (TXT/XML)...\n此区域将绘制真实缺陷位置以供比对。")
        self.lbl_gt_display.setStyleSheet("background-color: #f9f9fb; border: 1px dashed #bbb; color: #888; border-radius: 4px;")
        self.lbl_gt_display.setMinimumSize(250, 250)
        r_layout.addWidget(self.lbl_gt_display, stretch=2)
        
        top_splitter.addWidget(left_group); top_splitter.addWidget(right_group)
        top_splitter.setSizes([850, 490])
        
        # ====== 下半部分视窗 ======
        bottom_splitter = QSplitter(Qt.Horizontal)
        
        self.bottom_group = QGroupBox("多路模型性能协同推演视窗")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True); scroll.setMinimumHeight(280)
        self.scroll_content = QWidget()
        self.compare_layout = QHBoxLayout(self.scroll_content)
        scroll.setWidget(self.scroll_content)
        b_box = QVBoxLayout(self.bottom_group); b_box.addWidget(scroll)
        
        self.radar_group = QGroupBox("性能指标雷达图推演区")
        radar_layout = QVBoxLayout(self.radar_group)
        self.lbl_radar = ClickableLabel("等待评估指标输出...")
        self.lbl_radar.setAlignment(Qt.AlignCenter)
        self.lbl_radar.setStyleSheet("background-color: #f7f7f9; border: 1px solid #ccc; color: #888; cursor: pointer;")
        self.lbl_radar.setCursor(Qt.PointingHandCursor)
        self.lbl_radar.clicked.connect(self.show_large_preview)
        radar_layout.addWidget(self.lbl_radar)
        
        bottom_splitter.addWidget(self.bottom_group)
        bottom_splitter.addWidget(self.radar_group)
        bottom_splitter.setSizes([800, 540])
        
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(bottom_splitter)
        main_splitter.setSizes([550, 350])
        
        main_layout.addWidget(main_splitter)
        self.compare_labels = []

        self.btn_load_model.clicked.connect(self.select_models)
        self.btn_upload_img.clicked.connect(self.upload_image)
        self.btn_upload_vid.clicked.connect(self.upload_video)
        self.btn_stop.clicked.connect(self.stop_process)
        self.btn_history.clicked.connect(self.show_history_dialog)
        self.btn_upload_gt.clicked.connect(self.upload_gt_file)

    def select_models(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Load Weights", "", "Weights (*.pt)")
        if files:
            for f in files:
                if f not in self.compare_model_paths: self.compare_model_paths.append(f)
            self.lbl_model_info.setText(f"对比队列: {len(self.compare_model_paths)}")
            self._refresh_compare_ui()

    def _refresh_compare_ui(self):
        for l in self.compare_labels: l.parentWidget().deleteLater()
        self.compare_labels = []
        for p in self.compare_model_paths:
            container = QFrame()
            v_box = QVBoxLayout(container)
            lbl_img = ClickableLabel(); lbl_img.setFixedSize(380, 220)
            lbl_img.setAlignment(Qt.AlignCenter)
            lbl_img.setStyleSheet("background-color: #000; border: 1px solid #444;")
            lbl_img.setCursor(Qt.PointingHandCursor)
            lbl_img.clicked.connect(self.show_large_preview)
            title = QLabel(f"模型预测：{os.path.basename(p)}")
            title.setStyleSheet("font-weight: bold; color: #333;")
            v_box.addWidget(title); v_box.addWidget(lbl_img)
            self.compare_layout.addWidget(container); self.compare_labels.append(lbl_img)

    def log_message(self, msg):
        self.list_logs.addItem(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")
        self.list_logs.scrollToBottom()

    def upload_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Image", "", "Images (*.png *.jpg *.jpeg)")
        if path: 
            self.current_image_path = path
            self.start_analysis(path, 'image')

    def upload_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Video", "", "Videos (*.mp4 *.avi)")
        if path: self.start_analysis(path, 'video')

    def start_analysis(self, file_path, src_type):
        if self.inference_thread: 
            self.inference_thread.stop(); self.inference_thread.wait()
        self._refresh_compare_ui()
        self.inference_thread = YoloInferenceThread(file_path, src_type, self.compare_model_paths, parent=self)
        self.inference_thread.frame_ready.connect(self.on_main_stream)
        self.inference_thread.compare_frames_ready.connect(self.on_compare_stream)
        self.inference_thread.log_ready.connect(self.log_message)
        self.inference_thread.detection_finished.connect(lambda r: self.save_to_db(file_path, src_type, r))
        self.inference_thread.start()

    @pyqtSlot(QImage)
    def on_main_stream(self, img):
        pix = QPixmap.fromImage(img)
        self.lbl_display.setPixmapWithResize(pix)

    @pyqtSlot(list)
    def on_compare_stream(self, images):
        for i, img in enumerate(images):
            if i < len(self.compare_labels):
                lbl = self.compare_labels[i]
                lbl.setPixmapWithResize(QPixmap.fromImage(img))
        if self.inference_thread and self.inference_thread.source_type == 'image':
            os.makedirs("comparison_reports", exist_ok=True)
            path = f"comparison_reports/multi_snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.scroll_content.grab().save(path)
            self.log_message(f"Comparison report generated: {os.path.basename(path)}")

    def save_to_db(self, path, src_type, results):
        try:
            session = get_session()
            rec = InspectionRecord(inspector_name="Sys", source_type=src_type, file_path=path)
            session.add(rec); session.flush()
            rid = rec.id
            
            self.current_results = results
            for i, r in enumerate(results):
                det = DetectionResult(record_id=rid, class_name=r['class'], confidence=r['confidence'], bbox=str(r['bbox']))
                session.add(det)
            session.commit(); session.close()
            
            # 自动展示左侧同样的原图以作准备
            if hasattr(self, 'current_image_path') and self.current_image_path:
                self.lbl_gt_display.setPixmapWithResize(QPixmap(self.current_image_path))
            
            self.log_message(f"Archive Success (ID: {rid})")
            self.update_radar_ui()
        except Exception as e: self.log_message(f"DB Error: {e}")

    def update_radar_ui(self):
        radar_path = "comparison_reports/final_academic_comparison.png"
        if os.path.exists(radar_path):
            pix = QPixmap(radar_path)
            if not pix.isNull():
                self.lbl_radar.setPixmapWithResize(pix)

    def resizeEvent(self, event):
        # 移除了 lbl_radar 的手动缩放逻辑，完全交给 ImagePreviewLabel 类内部的 resizeEvent 处理。
        # 这样可以避免 MainWindow 与子控件之间产生尺寸反馈环路导致界面“无限膨胀”。
        super().resizeEvent(event)

    def stop_process(self):
        if self.inference_thread: self.inference_thread.stop()

    def show_history_dialog(self):
        dialog = QDialog(self); dialog.setWindowTitle("Historical Ledger"); dialog.resize(900, 500)
        table = QTableWidget(); layout = QVBoxLayout(dialog)
        table.setColumnCount(5); table.setHorizontalHeaderLabels(["ID", "Time", "Status", "Conf", "File"])
        try:
            session = get_session()
            recs = session.query(InspectionRecord).order_by(InspectionRecord.id.desc()).all()
            table.setRowCount(len(recs))
            for i, r in enumerate(recs):
                subs = session.query(DetectionResult).filter_by(record_id=r.id).all()
                stat, prob, is_def = "OK", "0.0%", False
                if subs:
                    defs = [s for s in subs if any(k in s.class_name.lower() for k in ['broken', 'defect'])]
                    t = sorted(defs, key=lambda x: x.confidence, reverse=True)[0] if defs else sorted(subs, key=lambda x: x.confidence, reverse=True)[0]
                    if defs: stat, is_def = "Fault", True
                    prob = f"{t.confidence*100:.1f}%"
                table.setItem(i, 0, QTableWidgetItem(str(r.id)))
                table.setItem(i, 1, QTableWidgetItem(str(r.inspection_time)))
                table.setItem(i, 2, QTableWidgetItem(stat))
                table.setItem(i, 3, QTableWidgetItem(prob))
                table.setItem(i, 4, QTableWidgetItem(r.file_path))
            session.close()
        except: pass
        layout.addWidget(table); dialog.exec_()

    def show_large_preview(self, label):
        if not hasattr(label, 'orig_pixmap') or label.orig_pixmap.isNull(): return
        dlg = QDialog(self)
        dlg.setWindowTitle("推演结果详情放大图")
        dlg.resize(1000, 750)
        l = QVBoxLayout(dlg)
        l.setContentsMargins(0, 0, 0, 0)
        
        img_lbl = ImagePreviewLabel()
        img_lbl.setPixmapWithResize(label.orig_pixmap)
        l.addWidget(img_lbl)
        dlg.exec_()

    def upload_gt_file(self):
        if not hasattr(self, 'current_results') or not self.current_results:
            self.log_message("ℹ️ 请先进行图片分析，再导入标准GT对比！")
            return
        
        path, _ = QFileDialog.getOpenFileName(self, "导入标注文件", "", "Label Files (*.txt *.xml)")
        if not path:
            return
            
        gt_boxes = []
        try:
            if path.endswith('.txt'):
                if not hasattr(self, 'current_image_path'):
                    self.log_message("⚠️ 缺少当前图像尺寸，无法解析比例坐标。")
                    return
                from PIL import Image
                img = Image.open(self.current_image_path)
                iw, ih = img.width, img.height
                with open(path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            x1 = (cx - w/2) * iw
                            y1 = (cy - h/2) * ih
                            x2 = (cx + w/2) * iw
                            y2 = (cy + h/2) * ih
                            gt_boxes.append({'class_id': cid, 'bbox': [x1, y1, x2, y2]})
                            
            elif path.endswith('.xml'):
                import xml.etree.ElementTree as ET
                tree = ET.parse(path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    bnd = obj.find("bndbox")
                    if bnd is not None:
                        x1 = float(bnd.find("xmin").text)
                        y1 = float(bnd.find("ymin").text)
                        x2 = float(bnd.find("xmax").text)
                        y2 = float(bnd.find("ymax").text)
                        gt_boxes.append({'class_name': obj.find("name").text, 'bbox': [x1, y1, x2, y2]})
                        
            self._draw_gt_boxes_and_show(gt_boxes)
        except Exception as e:
            self.log_message(f"⚠️ 解析GT文件失败: {e}")

    def _draw_gt_boxes_and_show(self, gt_boxes):
        import cv2
        import numpy as np
        if not hasattr(self, 'current_image_path'): return
        
        # Read the original image with support for Chinese paths
        img = cv2.imdecode(np.fromfile(self.current_image_path, dtype=np.uint8), -1)
        if img is None: 
            self.log_message("⚠️ 无法读取图像进行真值绘制。")
            return
            
        for gt in gt_boxes:
            box = gt['bbox']
            x1, y1, x2, y2 = map(int, box)
            cls_name = gt.get('class_name', f"Class {gt.get('class_id', '?')}")
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{cls_name} (GT)", (x1, max(y1-8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        r = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = r.shape
        qimg = QImage(r.data, w, h, c*w, QImage.Format_RGB888).copy()
        self.lbl_gt_display.setPixmapWithResize(QPixmap.fromImage(qimg))
        self.log_message(f"✅ 真值框可视化完毕！图上共绘制 {len(gt_boxes)} 个锚框。")
