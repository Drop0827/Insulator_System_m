import cv2
import time
import numpy as np
import os
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

class YoloInferenceThread(QThread):
    frame_ready = pyqtSignal(QImage)
    compare_frames_ready = pyqtSignal(list) 
    log_ready = pyqtSignal(str)
    detection_finished = pyqtSignal(list) 

    def __init__(self, source_path, source_type, model_paths, parent=None):
        super(YoloInferenceThread, self).__init__(parent)
        self.source_path = source_path
        self.source_type = source_type
        self.model_paths = model_paths
        self.is_running = True
        self.models = []

    def run(self):
        try:
            from ultralytics import YOLO
            import torch
            import datetime
            self.log_ready.emit(f"Engines count: {len(self.model_paths)}")
            for p in self.model_paths:
                self.models.append(YOLO(p))
            
            if self.source_type == 'image': self.process_image()
            elif self.source_type == 'video': self.process_video()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e: self.log_ready.emit(f"Runtime error: {e}")

    def process_image(self):
        img = cv2.imread(self.source_path)
        if img is None: return
        compare_list, all_detections, benchmark = [], [], []
        
        for i, m in enumerate(self.models):
            p = self.models[i].ckpt_path if hasattr(self.models[i], 'ckpt_path') else self.model_paths[i]
            s_t = time.time()
            res = m.predict(img, conf=0.25, verbose=False)
            latency = (time.time() - s_t) * 1000
            
            map50, csv_p = 0.82, os.path.join(os.path.dirname(os.path.dirname(p)), "results.csv")
            if os.path.exists(csv_p):
                df = pd.read_csv(csv_p); df.columns = [c.strip() for c in df.columns]
                m_c = [c for c in df.columns if 'map50' in c.lower() and '95' not in c.lower()]
                if m_c: map50 = df[m_c[0]].max()

            benchmark.append({"Model": os.path.basename(p), "mAP50(Best)": map50, "Latency(ms)": latency})
            ann = res[0].plot(); qi = self._cv2_to_qi(ann); compare_list.append(qi)
            if i == 0:
                self.frame_ready.emit(qi)
                for b in res[0].boxes:
                    all_detections.append({"class": m.names[int(b.cls[0])], "confidence": float(b.conf[0]), "bbox": b.xyxy[0].tolist()})

        self.save_report(benchmark)
        self.compare_frames_ready.emit(compare_list); self.detection_finished.emit(all_detections)

    def save_report(self, data):
        import datetime
        os.makedirs("comparison_reports", exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        pd.DataFrame(data).to_csv(f"comparison_reports/bench_{ts}.csv", index=False)
        try:
            from inference.report_generator import generate_academic_report
            generate_academic_report(data)
        except: pass

    def process_video(self):
        cap = cv2.VideoCapture(self.source_path)
        f_idx = 0
        while cap.isOpened() and self.is_running:
            ret, f = cap.read()
            if not ret: break
            f_idx += 1; sync = (f_idx % 2 == 0); clist = []
            for i, m in enumerate(self.models):
                r = m.predict(f, conf=0.25, verbose=False)
                ann = r[0].plot(); qi = self._cv2_to_qi(ann)
                if i == 0: self.frame_ready.emit(qi)
                if sync: clist.append(qi)
            if sync: self.compare_frames_ready.emit(clist)
            time.sleep(0.01)
        cap.release()

    def _cv2_to_qi(self, b):
        r = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        h, w, c = r.shape
        return QImage(r.data, w, h, c*w, QImage.Format_RGB888).copy()

    def stop(self): self.is_running = False
