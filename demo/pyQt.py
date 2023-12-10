from PyQt5.QtWidgets import QLabel, QPushButton
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
import mediapipe as mp
import sys
from PyQt5.QtGui import QFontDatabase, QFont
from predict.predict import preprocess, predict
import json
"""
export PYTHONPATH=export PYTHONPATH=/home/nastya/sign_language/GISLR/git/RecognitionofSignLanguage/utils:/home/nastya/sign_language/GISLR:/home/nastya/anaconda3/lib/python310.zip:/home/nastya/anaconda3/lib/python3.10:/home/nastya/anaconda3/lib/python3.10/lib-dynload:/home/nastya/.local/lib/python3.10/site-packages:/home/nastya/anaconda3/lib/python3.10/site-packages:/home/nastya/anaconda3/lib/python3.10/site-packages/PyQt5_sip-12.11.0-py3.10-linux-x86_64.egg:/home/nastya/anaconda3/lib/python3.10/site-packages/mpmath-1.2.1-py3.10.egg
"""

class VideoRecorder(QMainWindow):
    videoFinished = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Recorder")
        self.recording = False
        self.cap = None
        self.points = np.zeros([40, 543, 2])
        self.index = 0

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.test = None
        self.non_empty_frames_idxs = None
        self.y_pred = None
        self.show_points_b = False

        QFontDatabase.addApplicationFont("Gruppo_Regular.ttf")

        # Встановити шрифт
        font = QFont('Gruppo')
        font.setWeight(QFont.Bold)


        self.setStyleSheet("background-image: url('image/background_image.png'); background-repeat: no-repeat; background-position: center;")

        self.start_stop_button = QPushButton('Record video', self)
        self.start_stop_button.setFont(font)
        self.start_stop_button.clicked.connect(self.start_stop_video_recording)
        self.start_stop_button.resize(215, 53)
        self.start_stop_button.move(590, 530)
        self.start_stop_button.setStyleSheet("border-radius: 20px; background: #A1BCFF; font-size: 24px; font-weight: 400;")

        self.show_points = QPushButton('Show Points', self)
        self.show_points.setFont(font)
        self.show_points.clicked.connect(self.show_points_f)
        self.show_points.resize(215, 53)
        self.show_points.move(840, 530)
        self.show_points.setStyleSheet(
            '''
            QPushButton {
                border-radius: 20px;
                background: #A1BCFF;
                font-size: 24px;
                font-weight: 400;
            }
            QPushButton:pressed {
                background-color: #A9C6FF;
            }
            '''
        )

        self.predict = QPushButton('Predict', self)
        self.predict.setFont(font)
        self.predict.resize(215, 53)
        self.predict.move(130, 50)
        self.predict.setStyleSheet(
            '''
            QPushButton {
                border-radius: 20px;
                background: #CDF1EF;
                font-size: 24px;
                font-weight: 400;
            }
            QPushButton:pressed {
                background-color: #A9C6FF;
            }
            '''
        )
        #self.predict.setStyleSheet("border-radius: 20px; background: #CDF1EF; font-size: 24px; font-weight: 400;")
        self.predict.clicked.connect(self.display_result)

        self.result_label = QLabel(self)
        self.result_label.setFont(font)
        self.result_label.resize(180, 100)
        self.result_label.move(290, 185)
        self.result_label.setStyleSheet("background: transparent; font-size: 40px; font-weight: 800;")

        self.mp_holistic = mp.solutions.holistic  # Holistic model
        self.mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

        self.videoFinished.connect(self.enable_button)  # Підключення сигналу до слоту

        self.video_label = QLabel(self)
        self.video_label.resize(640, 480)
        self.video_label.move(490, 24)
        self.default_image = QPixmap('image/video_record.png')  # Зображення за замовчуванням
        self.video_label.setPixmap(self.default_image)


        # Кругова діаграма
        # Віджет для кругової діаграми
        self.pie = None
        self.ax = None
        self.history = ""
        self.diagram_widget = QWidget(self)
        self.diagram_widget.setGeometry(0, 80, 300, 300)  # Змініть розмір та положення за необхідності
        layout = QVBoxLayout(self.diagram_widget)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Встановлення стилю для прозорого фону без рамок
        self.diagram_widget.setStyleSheet("background: transparent;")
        self.figure.patch.set_facecolor('none')

        # Створення історії
        self.history_img = QLabel(self)
        self.history_img.resize(407, 190)
        self.history_img.move(38, 380)
        self.background_h = QPixmap('image/history.png')  # Зображення за замовчуванням
        self.history_img.setPixmap(self.background_h)

        self.history_label = QLabel("History", self)
        self.history_label.setFont(font)
        self.history_label.resize(135, 40)
        self.history_label.move(170, 390)
        self.history_label.setStyleSheet("background: transparent; font-size: 40px; font-weight: 800;")

        self.history_text = QLabel(self)
        self.history_text.setFont(font)
        self.history_text.resize(370, 150)
        self.history_text.move(55, 435)
        self.history_text.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.history_text.setWordWrap(True)
        self.history_text.setStyleSheet("background: transparent; font-size: 35px; font-weight: 400;")

    def draw_pie_chart(self):
        self.ax = self.figure.add_subplot(111)
        items = np.argsort(-self.y_pred, axis=1)[:, :5].flatten()
        self.labels = [self.get_sign(item) for item in items]
        self.sizes = self.y_pred[0][list(items)].flatten()# Приклад значень

        explode = [0.1 if s == max(self.sizes) else 0 for s in self.sizes]  # Виторкнення лише для найбільшого сегменту
        self.pie = self.ax.pie(self.sizes, startangle=0, explode = explode, colors=["#4E7EC6", "#10E58C", "#17044E", "#9454E7", "#40BCE3"])

        # Додати показ значень під час наведення миші
        self.tooltip = self.ax.annotate('', xy=(0, 0), xytext=(-100, 20), textcoords='offset points',
                                         bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.5),
                                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        self.tooltip.set_visible(False)

        self.ax.axis('equal')
        self.canvas.draw()
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)

    def on_hover(self, event):
        if event.inaxes == self.ax:
            for i, patch in enumerate(self.pie[0]):
                if patch.contains(event)[0]:
                    value = self.sizes[i]
                    percentage = (value / sum(self.sizes)) * 100

                    # Оновити текст підказки
                    self.tooltip.set_text(f"Значення: {self.labels[i]}\nВідсоток: {percentage:.2f}%")
                    self.tooltip.set_visible(True)
                    self.tooltip.xy = (event.xdata, event.ydata)
                    self.canvas.draw_idle()
                    break
            else:
                self.tooltip.set_visible(False)
                self.canvas.draw_idle()

    def show_points_f(self):
        if self.show_points_b:
            self.show_points_b = False
            self.show_points.setText('Show Points')
        else:
            self.show_points_b = True
            self.show_points.setText('Hide Points')

    def clear_pie_chart(self):
        if self.ax:
            self.ax.clear()
            self.ax.xaxis.set_visible(False)  # Приховати мітки на осі X
            self.ax.yaxis.set_visible(False)  # Приховати мітки на осі Y
            self.canvas.draw()

    def get_sign(self, result):

        def get_key_from_value(dictionary, search_value):
            for key, value in dictionary.items():
                if value == search_value:
                    return key
            return None

        with open(
                '/home/nastya/sign_language/GISLR/git/RecognitionofSignLanguage/preprocess_data/sign_to_prediction_index_map.json',
                'r') as file:
            # Load the JSON data
            data = json.load(file)

        return get_key_from_value(data, result)


    def display_result(self):
        self.clear_pie_chart()
        self.y_pred = predict(self.test, self.non_empty_frames_idxs)
        sign = self.get_sign(self.y_pred.argmax(axis=1))
        self.result_label.setText(f"Result: \n{sign}")
        self.draw_pie_chart()
        self.history += sign + " "
        self.history_text.setText(self.history)

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_styled_landmarks(self, image, results):
        if self.show_points_b:
            # Draw face connections
            self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                                      self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                      )
            # Draw pose connections
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                      self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )
            # Draw left hand connections
            self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                      self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                      )
            # Draw right hand connections
            self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                      self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

    def get_coordination_without_z(self, results):
        face = np.array([[res.x, res.y] for res in results.face_landmarks.landmark])
        pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark])
        lh = np.array([[res.x, res.y] for res in
                       results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.array(
            [[0.0, 0.0] for res in range(21)])
        rh = np.array([[res.x, res.y] for res in
                       results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.array(
            [[0.0, 0.0] for res in range(21)])
        return np.concatenate((face, lh, pose, rh))


    def enable_button(self):
        self.recording = False  # Встановлення прапорця запису в False після завершення запису
        self.start_stop_button.setEnabled(True)  # Розблокувати кнопку після завершення зйомки
        self.video_label.setPixmap(self.default_image)
        self.test, self.non_empty_frames_idxs = preprocess(self.points)

    def start_video_stream(self):
        if not self.recording:  # Перевірка, чи не йде вже запис
            self.recording = True
            self.start_stop_button.setEnabled(False)  # Блокування кнопки під час запису
            self.cap = cv2.VideoCapture(0)
            self.index = 0
            self.show_frame()

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            image, results = self.mediapipe_detection(frame, self.holistic)
            self.draw_styled_landmarks(image, results)

            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(640, 480)
            self.video_label.setPixmap(pixmap)

            current_points = self.get_coordination_without_z(results)
            self.points[self.index] = current_points
            self.index += 1

            if self.index < 40:
                QTimer.singleShot(10, self.show_frame)
            else:
                self.cap.release()
                self.videoFinished.emit()  # Відправити сигнал про завершення відеозапису

    def start_stop_video_recording(self):
        self.start_video_stream()  # Запустити або зупинити запис відео при натисканні кнопки

def main():
    app = QApplication(sys.argv)
    window = VideoRecorder()
    window.setGeometry(100, 100, 1150, 630)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
