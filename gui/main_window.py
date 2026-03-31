import sys
import os
import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QListWidget, QGroupBox, QSplitter,
    QTableWidget, QTableWidgetItem, QDialog, QMessageBox, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage

from inference.detector_thread import YoloInferenceThread
from db.models import get_session, InspectionRecord, DetectionResult

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("输电线路绝缘子故障智能诊断与性能推演平台 — 实验版本")
        self.resize(1340, 950)

        p_best = r"C:\Develop\BS\Insulator_System_m\runs\train\yolo11_WTConv_exp\weights\best.pt"
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
        
        splitter = QSplitter(Qt.Horizontal)
        left_group = QGroupBox("诊断视图")
        self.lbl_display = QLabel("Waiting for input...")
        self.lbl_display.setAlignment(Qt.AlignCenter)
        self.lbl_display.setMinimumSize(800, 500)
        self.lbl_display.setStyleSheet("background-color: #f7f7f9; border: 1px solid #ccc;")
        l_layout = QVBoxLayout(left_group); l_layout.addWidget(self.lbl_display)
        
        right_group = QGroupBox("诊断日志输出")
        self.list_logs = QListWidget()
        r_layout = QVBoxLayout(right_group); r_layout.addWidget(self.list_logs)
        
        splitter.addWidget(left_group); splitter.addWidget(right_group)
        splitter.setSizes([850, 490])
        main_layout.addWidget(splitter)
        
        self.bottom_group = QGroupBox("多路模型性能协同推演视窗")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True); scroll.setFixedHeight(230)
        self.scroll_content = QWidget()
        self.compare_layout = QHBoxLayout(self.scroll_content)
        scroll.setWidget(self.scroll_content)
        b_box = QVBoxLayout(self.bottom_group); b_box.addWidget(scroll)
        main_layout.addWidget(self.bottom_group)
        self.compare_labels = []

        self.btn_load_model.clicked.connect(self.select_models)
        self.btn_upload_img.clicked.connect(self.upload_image)
        self.btn_upload_vid.clicked.connect(self.upload_video)
        self.btn_stop.clicked.connect(self.stop_process)
        self.btn_history.clicked.connect(self.show_history_dialog)

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
            lbl_img = QLabel(); lbl_img.setFixedSize(300, 170)
            lbl_img.setStyleSheet("background-color: #000; border: 1px solid #444;")
            v_box.addWidget(QLabel(os.path.basename(p))); v_box.addWidget(lbl_img)
            self.compare_layout.addWidget(container); self.compare_labels.append(lbl_img)

    def log_message(self, msg):
        self.list_logs.addItem(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")
        self.list_logs.scrollToBottom()

    def upload_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Image", "", "Images (*.png *.jpg *.jpeg)")
        if path: self.start_analysis(path, 'image')

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
        self.lbl_display.setPixmap(pix.scaled(self.lbl_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @pyqtSlot(list)
    def on_compare_stream(self, images):
        for i, img in enumerate(images):
            if i < len(self.compare_labels):
                lbl = self.compare_labels[i]
                lbl.setPixmap(QPixmap.fromImage(img).scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
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
            for r in results:
                det = DetectionResult(record_id=rid, class_name=r['class'], confidence=r['confidence'], bbox=str(r['bbox']))
                session.add(det)
            session.commit(); session.close()
            self.log_message(f"Archive Success (ID: {rid})")
        except Exception as e: self.log_message(f"DB Error: {e}")

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
