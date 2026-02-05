import sys
import os
import csv
import time
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.ops import nms
import mediapipe as mp
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QFrame, QCheckBox,
    QComboBox, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon

if sys.platform == "win32":
    BACKENDS = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto")
    ]
elif sys.platform == "darwin":
    BACKENDS = [
        (cv2.CAP_AVFOUNDATION, "AVFoundation"),
        (cv2.CAP_ANY, "Auto")
    ]
else:
    BACKENDS = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY, "Auto")
    ]

RESOLUTIONS = {
    "720p (1280×720)": (1280, 720),
    "480p (854×480)": (854, 480),
    "360p (640×360)": (640, 360)
}

NOSE_THRESHOLD = 0.25
PAD_RATIO = 0.25
NOSE = 1
LEFT_EYE = 130
RIGHT_EYE = 359
LOG_MAX_LINES = 100
MIN_DISENGAGEMENT_DURATION = 2.0
DISCONNECTION_WINDOW = 600


class EngagementStatsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.total_students = 0
        self.engaged_count = 0
        self.engagement_ratio = 0.0
        self.class_status = "Нет учеников"
        self.max_disengaged_10min = 0
        self.setFixedHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def update_stats(self, total, engaged):
        self.total_students = total
        self.engaged_count = engaged
        self.engagement_ratio = engaged / total if total > 0 else 0.0

        if total == 0:
            self.class_status = "Нет учеников"
        elif self.engagement_ratio >= 0.8:
            self.class_status = "Класс вовлечён"
        elif self.engagement_ratio >= 0.3:
            self.class_status = "Частично вовлечён"
        else:
            self.class_status = "Класс НЕ вовлечён"
        self.update()

    def update_disengagement_summary(self, max_disengaged):
        self.max_disengaged_10min = max_disengaged
        self.update()

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QPen, QBrush
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bg_color = self.palette().color(QPalette.Window)
        text_color = self.palette().color(QPalette.WindowText)
        painter.fillRect(self.rect(), bg_color)

        font = QFont("Segoe UI", 11)
        painter.setFont(font)
        painter.setPen(text_color)
        info_text = f"Учеников: {self.total_students} | Вовлечено: {self.engaged_count}"
        status_text = f"Статус: {self.class_status}"
        disengaged_text = f"Макс. отвлечено за 10 мин: {self.max_disengaged_10min}"
        painter.drawText(20, 30, info_text)
        painter.drawText(20, 60, status_text)
        painter.drawText(20, 90, disengaged_text)

        start_x = 500
        bar_width = 160
        bar_height = 8
        bar_y = 22

        is_dark = bg_color.lightness() < 150
        frame_color = QColor(80, 80, 80) if is_dark else QColor(200, 200, 200)
        fill_bg_color = QColor(50, 50, 50) if is_dark else QColor(230, 230, 230)

        painter.setPen(QPen(frame_color, 1))
        painter.setBrush(QBrush(fill_bg_color))
        painter.drawRect(start_x, bar_y, bar_width, bar_height)

        if self.total_students == 0:
            fill_color = QColor(120, 120, 120)
        elif self.engagement_ratio >= 0.8:
            fill_color = QColor(46, 204, 113)
        elif self.engagement_ratio >= 0.3:
            fill_color = QColor(52, 152, 219)
        else:
            fill_color = QColor(231, 76, 60)

        fill_w = int(bar_width * self.engagement_ratio)
        painter.setBrush(QBrush(fill_color))
        painter.drawRect(start_x, bar_y, fill_w, bar_height)

        percent_text = f"{self.engagement_ratio:.0%}"
        percent_font = QFont("Segoe UI", 10, QFont.Bold)
        painter.setFont(percent_font)
        painter.setPen(text_color)
        painter.drawText(start_x + bar_width + 10, 30, percent_text)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    log_signal = pyqtSignal(str)
    stats_signal = pyqtSignal(int, int)
    disengagement_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, camera_index=0, show_mesh=False, backend=cv2.CAP_ANY, resolution=(1280, 720)):
        super().__init__()
        self.camera_index = camera_index
        self.show_mesh = show_mesh
        self.backend = backend
        self.resolution = resolution
        self._run_flag = False
        self.engagement_state = {}
        self.fps = 10

    def run(self):
        model_path = Path("yolov8n.pt")
        if not model_path.exists():
            self.log_signal.emit(f"❌ Файл модели не найден: {model_path.absolute()}")
            self.finished_signal.emit()
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            yolo = YOLO(str(model_path))
            yolo.to(device)
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8
            )
        except Exception as e:
            self.log_signal.emit(f"❌ Ошибка загрузки модели: {str(e)}")
            self.finished_signal.emit()
            return

        self.log_signal.emit("✅ Модели загружены. Запуск камеры...")
        cap = cv2.VideoCapture(self.camera_index, self.backend)
        if not cap.isOpened():
            self.log_signal.emit("❌ Камера недоступна. Закройте другие программы.")
            self.finished_signal.emit()
            return

        w, h = self.resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self._run_flag = True
        self.log_signal.emit(f"📹 Анализ начат ({w}×{h})")
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        MESH_STYLE = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()

        frame_count = 0
        start_time = time.time()
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                self.log_signal.emit("⚠️ Камера отключена. Остановка.")
                break

            annotated = frame.copy()

            det_results = yolo.predict(frame, classes=[0], verbose=False)
            if det_results[0].boxes is not None and len(det_results[0].boxes) > 0:
                boxes_xyxy = det_results[0].boxes.xyxy.cpu()
                confidences = det_results[0].boxes.conf.cpu()
                keep_indices = nms(boxes_xyxy, confidences, iou_threshold=0.5)
                filtered_boxes = boxes_xyxy[keep_indices]
                results = yolo.track(frame, persist=True, classes=[0], tracker="bytetrack_classroom.yaml",
                                     boxes=filtered_boxes)
            else:
                results = yolo.track(frame, persist=True, classes=[0], tracker="bytetrack_classroom.yaml")

            active_ids = set()
            class_engaged_count = 0
            disengaged_count = 0

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().int().numpy()
                ids = results[0].boxes.id.int().cpu().numpy()

                valid_mask = ids <= 50
                boxes = boxes[valid_mask]
                ids = ids[valid_mask]

                if frame_count % 10 == 0 and frame_count > 0:
                    elapsed = time.time() - start_time
                    self.fps = 10 / elapsed if elapsed > 0 else 10
                    start_time = time.time()

                for (x1, y1, x2, y2), track_id in zip(boxes, ids):
                    track_id = int(track_id)
                    active_ids.add(track_id)
                    pad = int(PAD_RATIO * (y2 - y1))
                    y1_crop = max(0, y1 - pad)
                    y2_crop = min(frame.shape[0], y2 + pad)
                    x1_crop = max(0, x1 - pad)
                    x2_crop = min(frame.shape[1], x2 + pad)

                    face = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                    currently_engaged = False

                    if face.size > 0:
                        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face_res = face_mesh.process(rgb)

                        if face_res.multi_face_landmarks and self.show_mesh:
                            for face_landmarks in face_res.multi_face_landmarks:
                                original_landmarks = []
                                for lm in face_landmarks.landmark:
                                    original_landmarks.append((lm.x, lm.y, lm.z))
                                    lm.x = (lm.x * (x2_crop - x1_crop) + x1_crop) / frame.shape[1]
                                    lm.y = (lm.y * (y2_crop - y1_crop) + y1_crop) / frame.shape[0]

                                mp_drawing.draw_landmarks(
                                    image=annotated,
                                    landmark_list=face_landmarks,
                                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=MESH_STYLE
                                )

                                for lm, (x, y, z) in zip(face_landmarks.landmark, original_landmarks):
                                    lm.x, lm.y, lm.z = x, y, z

                        if face_res.multi_face_landmarks:
                            lm = face_res.multi_face_landmarks[0].landmark
                            if 0.05 < lm[LEFT_EYE].x < 0.95 and 0.05 < lm[RIGHT_EYE].x < 0.95:
                                h_face, w_face = face.shape[:2]
                                nose_x = lm[NOSE].x * w_face
                                left_x = lm[LEFT_EYE].x * w_face
                                right_x = lm[RIGHT_EYE].x * w_face
                                center = (left_x + right_x) / 2
                                width = abs(right_x - left_x) + 1e-5
                                offset = abs(nose_x - center) / width
                                currently_engaged = offset <= NOSE_THRESHOLD

                    if track_id not in self.engagement_state:
                        self.engagement_state[track_id] = {
                            'engaged': currently_engaged,
                            'start_time': time.time(),
                            'marked_disengaged': not currently_engaged
                        }

                    state = self.engagement_state[track_id]
                    current_time = time.time()

                    if state['engaged'] != currently_engaged:
                        state['engaged'] = currently_engaged
                        state['start_time'] = current_time
                        if currently_engaged:
                            state['marked_disengaged'] = False

                    if not currently_engaged:
                        duration = current_time - state['start_time']
                        if duration >= MIN_DISENGAGEMENT_DURATION:
                            state['marked_disengaged'] = True
                        else:
                            state['marked_disengaged'] = False
                    else:
                        state['marked_disengaged'] = False

                    is_disengaged_long = state['marked_disengaged']
                    color = (0, 0, 255) if is_disengaged_long else (0, 255, 0)
                    if not is_disengaged_long:
                        class_engaged_count += 1
                    else:
                        disengaged_count += 1

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f'ID:{track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            self.engagement_state = {k: v for k, v in self.engagement_state.items() if k in active_ids}

            total_students = len(active_ids)
            self.stats_signal.emit(total_students, class_engaged_count)
            self.disengagement_signal.emit(disengaged_count)

            cv2.putText(annotated, f'FPS: {self.fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            self.change_pixmap_signal.emit(annotated)

            frame_count += 1

        cap.release()
        if 'face_mesh' in locals():
            face_mesh.close()
        self.finished_signal.emit()

    def stop(self):
        self._run_flag = False
        self.wait()


class EngageTrackApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EngageTrack — Анализ вовлечённости")
        self.resize(1150, 850)
        self.is_dark_mode = False
        self.disengagement_log = deque(maxlen=600)
        self.analysis_start_time = None
        self.total_frames = 0
        self.cumulative_students = 0
        self.cumulative_engaged = 0
        self.max_disengaged_overall = 0
        self.engagement_history = []

        if os.path.exists("engagetrack.ico"):
            self.setWindowIcon(QIcon("engagetrack.ico"))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        title_layout = QHBoxLayout()
        title_label = QLabel("EngageTrack")
        title_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title_layout.addWidget(title_label)

        self.theme_button = QPushButton("🌙 Тёмная тема")
        self.theme_button.setFixedWidth(140)
        self.theme_button.clicked.connect(self.toggle_theme)
        title_layout.addStretch()
        title_layout.addWidget(self.theme_button)
        main_layout.addLayout(title_layout)

        subtitle = QLabel("Анализ вовлечённости в реальном времени")
        subtitle.setFont(QFont("Segoe UI", 11))
        main_layout.addWidget(subtitle)

        privacy_label = QLabel("ℹ️ Система анализирует только позу головы. Изображения не сохраняются.")
        privacy_label.setObjectName("privacy_label")
        privacy_label.setFont(QFont("Segoe UI", 9))
        privacy_label.setWordWrap(True)
        main_layout.addWidget(privacy_label)

        control_frame = QFrame()
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(15, 10, 15, 10)

        cam_layout = QHBoxLayout()
        cam_layout.addWidget(QLabel("Камера:"))
        self.camera_combo = QComboBox()
        cam_layout.addWidget(self.camera_combo)
        self.refresh_cam_button = QPushButton("🔄 Обновить")
        self.refresh_cam_button.setFixedWidth(100)
        self.refresh_cam_button.clicked.connect(self.detect_cameras)
        cam_layout.addWidget(self.refresh_cam_button)
        control_layout.addLayout(cam_layout)

        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Разрешение:"))
        self.resolution_combo = QComboBox()
        for name in RESOLUTIONS.keys():
            self.resolution_combo.addItem(name)
        self.resolution_combo.setCurrentText("720p (1280×720)")
        res_layout.addWidget(self.resolution_combo)
        control_layout.addLayout(res_layout)

        self.mesh_checkbox = QCheckBox("Показывать меш лица")
        self.mesh_checkbox.setChecked(False)
        control_layout.addWidget(self.mesh_checkbox)

        self.start_button = QPushButton("▶ Начать анализ")
        self.stop_button = QPushButton("⏹ Остановить")
        self.stop_button.setEnabled(False)
        self.export_button = QPushButton("📄 Экспорт отчёта")
        self.export_button.setEnabled(False)

        control_layout.addStretch()
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.export_button)
        control_frame.setLayout(control_layout)
        main_layout.addWidget(control_frame)

        self.stats_widget = EngagementStatsWidget()
        main_layout.addWidget(self.stats_widget)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 450)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setText("Нажмите «Начать анализ» для запуска")
        main_layout.addWidget(self.video_label)

        log_frame = QFrame()
        log_layout = QVBoxLayout()
        log_label = QLabel("Журнал событий")
        log_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMaximumHeight(100)
        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log_text)
        log_frame.setLayout(log_layout)
        main_layout.addWidget(log_frame)

        central_widget.setLayout(main_layout)

        self.start_button.clicked.connect(self.start_video)
        self.stop_button.clicked.connect(self.stop_video)
        self.export_button.clicked.connect(self.export_report)

        self.thread = None
        self.session_log = []
        self.detect_cameras()
        self.apply_theme()

        self.log("ℹ️ EngageTrack запущен. Нажмите «🔄 Обновить», если подключили новую камеру.")

    def detect_cameras(self):
        self.refresh_cam_button.setEnabled(False)
        self.refresh_cam_button.setText("🔍 Сканирование...")

        current_text = self.camera_combo.currentText()
        self.camera_combo.clear()

        found_cameras = []
        max_test_index = 5

        for cam_id in range(max_test_index):
            cap = None
            try:
                backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
                cap = cv2.VideoCapture(cam_id, backend)
                if not cap.isOpened():
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                frame = None
                for _ in range(2):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        break
                    time.sleep(0.1)

                if frame is not None and frame.size > 0:
                    name = f"Камера {cam_id}"
                    found_cameras.append((cam_id, name))
                cap.release()
                time.sleep(0.05)
            except Exception:
                pass
            finally:
                if cap and cap.isOpened():
                    cap.release()

        if found_cameras:
            for cam_id, label in found_cameras:
                self.camera_combo.addItem(label, cam_id)
            for i in range(self.camera_combo.count()):
                if self.camera_combo.itemText(i) == current_text:
                    self.camera_combo.setCurrentIndex(i)
                    break
        else:
            self.camera_combo.addItem("Камеры не найдены", -1)

        self.refresh_cam_button.setText("🔄 Обновить")
        self.refresh_cam_button.setEnabled(True)

    def apply_theme(self):
        if self.is_dark_mode:
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(30, 33, 36))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(25, 28, 31))
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(50, 54, 57))
            palette.setColor(QPalette.ButtonText, Qt.white)
            self.setPalette(palette)

            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #1e2124; color: white; }
                QFrame { background-color: #25282b; border-radius: 12px; }
                QTextEdit { background-color: #1e2225; color: #e0e0e0; border: 1px solid #444; border-radius: 6px; font-family: Consolas; font-size: 9pt; }
                QLabel { color: white; }
                QComboBox, QCheckBox { background-color: #2d3034; color: white; border: 1px solid #555; padding: 4px; }
                QPushButton {
                    background-color: #3a3d41; color: white; border: 1px solid #555;
                    border-radius: 8px; padding: 6px 12px; font-weight: bold;
                }
                QPushButton:hover { background-color: #4a4d51; }
                QPushButton:disabled { background-color: #3a3d41; color: #777; }
            """)
            privacy_label = self.findChild(QLabel, "privacy_label")
            if privacy_label:
                privacy_label.setStyleSheet("color: #f1c40f;")
            self.theme_button.setText("☀️ Светлая тема")
        else:
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(245, 247, 250))
            palette.setColor(QPalette.WindowText, Qt.black)
            palette.setColor(QPalette.Base, Qt.white)
            palette.setColor(QPalette.Text, Qt.black)
            palette.setColor(QPalette.Button, Qt.white)
            palette.setColor(QPalette.ButtonText, Qt.black)
            self.setPalette(palette)

            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #f5f7fa; color: black; }
                QFrame { background-color: white; border-radius: 12px; }
                QTextEdit { background-color: #fdfdfd; color: #333; border: 1px solid #bdc3c7; border-radius: 6px; font-family: Consolas; font-size: 9pt; }
                QLabel { color: black; }
                QComboBox, QCheckBox { background-color: white; color: black; border: 1px solid #ccc; padding: 4px; }
                QPushButton {
                    background-color: #ecf0f1; color: black; border: 1px solid #bdc3c7;
                    border-radius: 8px; padding: 6px 12px; font-weight: bold;
                }
                QPushButton:hover { background-color: #d5dbdb; }
                QPushButton:disabled { background-color: #ecf0f1; color: #95a5a6; }
            """)
            privacy_label = self.findChild(QLabel, "privacy_label")
            if privacy_label:
                privacy_label.setStyleSheet("color: #e74c3c;")
            self.theme_button.setText("🌙 Тёмная тема")

        self.update()

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme()

    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        self.session_log.append((timestamp, msg))
        self.log_text.append(full_msg)
        if self.log_text.document().blockCount() > LOG_MAX_LINES:
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def start_video(self):
        cam_index = self.camera_combo.currentData()
        if cam_index == -1:
            QMessageBox.warning(self, "Нет камеры", "Камера недоступна.")
            return

        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        res_name = self.resolution_combo.currentText()
        resolution = RESOLUTIONS[res_name]

        self.analysis_start_time = time.time()
        self.total_frames = 0
        self.cumulative_students = 0
        self.cumulative_engaged = 0
        self.max_disengaged_overall = 0
        self.engagement_history = []

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.video_label.setText("Загрузка моделей ИИ... Подождите.")
        self.log("⏳ Запуск...")

        self.thread = VideoThread(
            camera_index=cam_index,
            show_mesh=self.mesh_checkbox.isChecked(),
            backend=backend,
            resolution=resolution
        )
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.log_signal.connect(self.log)
        self.thread.stats_signal.connect(self.on_student_stats_update)
        self.thread.disengagement_signal.connect(self.log_disengagement)
        self.thread.finished_signal.connect(self.on_thread_finished)
        self.thread.start()

    def on_student_stats_update(self, total, engaged):
        self.stats_widget.update_stats(total, engaged)
        if self.analysis_start_time is not None:
            self.total_frames += 1
            self.cumulative_students += total
            self.cumulative_engaged += engaged
            ratio = engaged / total if total > 0 else 0.0
            self.engagement_history.append(ratio)

    def log_disengagement(self, disengaged_count):
        self.disengagement_log.append((time.time(), disengaged_count))
        current_time = time.time()
        recent_values = [
            count for ts, count in self.disengagement_log
            if current_time - ts <= DISCONNECTION_WINDOW
        ]
        max_disengaged_10min = max(recent_values) if recent_values else 0
        self.stats_widget.update_disengagement_summary(max_disengaged_10min)

        if disengaged_count > self.max_disengaged_overall:
            self.max_disengaged_overall = disengaged_count

    def stop_video(self):
        if self.thread:
            self.thread.stop()
        self.on_thread_finished()

    def on_thread_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.export_button.setEnabled(True)
        self.video_label.setText("Нажмите «Начать анализ» для запуска")
        self.stats_widget.update_stats(0, 0)
        self.stats_widget.update_disengagement_summary(0)
        self.log("⏹ Анализ остановлен.")

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def export_report(self):
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        csv_filename = f"engagetrack_report_{timestamp_str}.csv"
        plot_filename = f"engagement_plot_{timestamp_str}.png"

        try:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["ТИП", "ЗНАЧЕНИЕ", "ЕД. ИЗМ."])

                if self.analysis_start_time is not None:
                    duration_sec = time.time() - self.analysis_start_time
                    duration_min = duration_sec / 60
                    writer.writerow(["Длительность анализа", f"{duration_min:.1f}", "мин"])
                else:
                    writer.writerow(["Длительность анализа", "0.0", "мин"])

                if self.total_frames > 0:
                    avg_students = self.cumulative_students / self.total_frames
                    avg_engaged = self.cumulative_engaged / self.total_frames
                    avg_engagement_pct = (avg_engaged / avg_students * 100) if avg_students > 0 else 0.0
                    writer.writerow(["Среднее число учеников", f"{avg_students:.1f}", "чел."])
                    writer.writerow(["Средняя вовлечённость", f"{avg_engagement_pct:.1f}", "%"])
                else:
                    writer.writerow(["Среднее число учеников", "0.0", "чел."])
                    writer.writerow(["Средняя вовлечённость", "0.0", "%"])

                writer.writerow(["Макс. отвлечено за урок", str(self.max_disengaged_overall), "чел."])
                writer.writerow([])

                error_entries = [
                    (ts, msg) for ts, msg in self.session_log
                    if "❌" in msg or "⚠️" in msg
                ]

                if error_entries:
                    writer.writerow(["ВРЕМЯ", "СОБЫТИЕ"])
                    for ts, msg in error_entries:
                        writer.writerow([ts, msg])
                else:
                    writer.writerow(["Ошибки и предупреждения отсутствуют"])

            if self.engagement_history:
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    from scipy.ndimage import uniform_filter1d

                    y = np.array(self.engagement_history)
                    x = np.arange(len(y))

                    window_size = max(1, len(y) // 20)
                    if window_size > 1:
                        y_smooth = uniform_filter1d(y, size=window_size, mode='nearest')
                    else:
                        y_smooth = y

                    plt.figure(figsize=(10, 4))
                    plt.plot(x, y_smooth * 100, color='#2ecc71', linewidth=1.8, label='Вовлечённость')
                    plt.axhline(80, color='#2ecc71', linestyle='--', alpha=0.6, linewidth=1)
                    plt.axhline(30, color='#3498db', linestyle='--', alpha=0.6, linewidth=1)
                    plt.fill_between(x, 80, 100, color='#2ecc71', alpha=0.1)
                    plt.fill_between(x, 30, 80, color='#3498db', alpha=0.1)
                    plt.fill_between(x, 0, 30, color='#e74c3c', alpha=0.1)
                    plt.ylim(0, 100)
                    plt.xlabel('Кадр')
                    plt.ylabel('Вовлечённость (%)')
                    plt.title('Динамика вовлечённости класса во время урока')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(plot_filename, dpi=150)
                    plt.close()
                except Exception as e:
                    self.log(f"⚠️ Не удалось сохранить график: {e}")

            QMessageBox.information(self, "Экспорт завершён",
                                    f"Отчёт и график сохранены:\n{os.path.abspath(csv_filename)}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", f"Не удалось сохранить отчёт:\n{str(e)}")

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = EngageTrackApp()
    window.show()
    sys.exit(app.exec_())
