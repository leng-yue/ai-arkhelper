import ctypes
import time
from hashlib import md5
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
from adbutils import adb
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

import scrcpy

from base import ACTIONS
from predict import Predictor
from ui_main import Ui_MainWindow

app = QApplication([])


class MainWindow(QMainWindow):
    def __init__(self, max_width: Optional[int], serial: Optional[str] = None):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Setup devices
        self.devices = self.list_devices()
        if serial:
            self.choose_device(serial)
        self.device = adb.device(serial=self.ui.combo_device.currentText())
        self.alive = True

        # Setup client
        self.client = scrcpy.Client(
            max_width=max_width, device=self.device, max_fps=10,
            lock_screen_orientation=scrcpy.LOCK_SCREEN_ORIENTATION_0
        )
        self.client.add_listener(scrcpy.EVENT_INIT, self.on_init)
        self.client.add_listener(scrcpy.EVENT_FRAME, self.on_frame)

        # Bind controllers
        self.ui.button_home.clicked.connect(self.on_click_home)
        self.ui.button_back.clicked.connect(self.on_click_back)
        self.ui.button_save.clicked.connect(self.on_click_save)

        # Bind config
        self.ui.combo_device.currentTextChanged.connect(self.choose_device)
        self.ui.list_task_types.addItems(ACTIONS)

        # Bind mouse event
        self.ui.label.mousePressEvent = self.on_mouse_event(scrcpy.ACTION_DOWN)
        self.ui.label.mouseMoveEvent = self.on_mouse_event(scrcpy.ACTION_MOVE)
        self.ui.label.mouseReleaseEvent = self.on_mouse_event(scrcpy.ACTION_UP)

        # Label tool
        self.cropping: Optional[tuple[tuple[int, int], tuple[int, int]]] = None
        self.frame: Optional[np.ndarray] = None
        self.predictor = Predictor()
        self.last_click = time.time()

    def choose_device(self, device):
        if device not in self.devices:
            msgBox = QMessageBox()
            msgBox.setText(f"Device serial [{device}] not found!")
            msgBox.exec()
            return

        # Ensure text
        self.ui.combo_device.setCurrentText(device)
        # Restart service
        if getattr(self, "client", None):
            self.client.stop()
            self.client.device = adb.device(serial=device)

    def list_devices(self):
        self.ui.combo_device.clear()
        items = [i.serial for i in adb.device_list()]
        self.ui.combo_device.addItems(items)
        return items

    def on_click_home(self):
        self.client.control.keycode(scrcpy.KEYCODE_HOME, scrcpy.ACTION_DOWN)
        self.client.control.keycode(scrcpy.KEYCODE_HOME, scrcpy.ACTION_UP)

    def on_click_back(self):
        self.client.control.back_or_turn_screen_on(scrcpy.ACTION_DOWN)
        self.client.control.back_or_turn_screen_on(scrcpy.ACTION_UP)

    def on_click_save(self):
        flag = "?????????" if self.ui.check_task_finished.isChecked() else "?????????"
        self.__save_img(flag)

    def __save_img(self, filename: str):
        items = self.ui.list_task_types.selectedItems()
        if len(items) != 1:
            ctypes.windll.user32.MessageBoxW(0, "??????????????????!", "??????", 0)
            return

        base_path = Path("records") / items[0].text()
        if not base_path.exists():
            base_path.mkdir(parents=True)
        _, result = cv2.imencode(".jpg", self.frame)
        path = base_path / f"{md5(result).hexdigest()}_{filename}.jpg"
        path.write_bytes(result)
        print("???????????????", path)

    def on_mouse_event(self, action=scrcpy.ACTION_DOWN):
        def handler(evt: QMouseEvent):
            x, y = int(evt.position().x()), int(evt.position().y())
            if action == scrcpy.ACTION_DOWN:
                self.cropping = (x, y), (x, y)
            elif self.cropping is not None and action == scrcpy.ACTION_MOVE:
                self.cropping = self.cropping[0], (x, y)
            elif self.cropping is not None and action == scrcpy.ACTION_UP:
                cropping = self.cropping
                self.cropping = None
                # if ctypes.windll.user32.MessageBoxW(0, "????????????????????????????", "??????", 1) == 2:
                #     return

                # ????????????
                x0, y0, x1, y1 = cropping[0] + cropping[1]
                if abs(x1 - x0) < 20 or abs(y1 - y0) < 20:
                    # ??????
                    self.client.control.touch((x0 + x1) / 2, (y0 + y1) / 2, scrcpy.ACTION_DOWN)
                    self.client.control.touch((x0 + x1) / 2, (y0 + y1) / 2, scrcpy.ACTION_UP)
                    return

                flag = "?????????" if self.ui.check_task_finished.isChecked() else "?????????"
                box = ','.join(map(str, cropping[0] + cropping[1]))
                self.__save_img(f"{flag}_{box}")

        return handler

    def on_init(self):
        self.setWindowTitle(f"Serial: {self.client.device_name}")

    def on_frame(self, frame):
        app.processEvents()
        if frame is None:
            return

        self.frame = frame.copy()
        if self.cropping is not None:
            cv2.rectangle(frame, self.cropping[0], self.cropping[1], (0, 0, 255), 2)

        items = self.ui.list_task_types.selectedItems()
        if len(items) == 1:
            finished, score, (x, y) = self.predictor.predict(self.frame, items[0].text())
            self.setWindowTitle(f"Serial: {self.client.device_name}, Finished: {finished}")
            if finished:
                self.ui.check_execute.setChecked(False)

            if score > 0.2:
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                if self.ui.check_execute.isChecked() and self.last_click < time.time() - 0.5:
                    self.last_click = time.time()
                    print("CLICKED", self.last_click)
                    self.client.control.touch(x, y, scrcpy.ACTION_DOWN)
                    self.client.control.touch(x, y, scrcpy.ACTION_UP)

        image = QImage(
            frame,
            frame.shape[1],
            frame.shape[0],
            frame.shape[1] * 3,
            QImage.Format_BGR888,
        )
        pix = QPixmap(image)
        self.ui.label.setPixmap(pix)
        self.resize(*self.client.resolution)

    def closeEvent(self, _):
        self.client.stop()
        self.alive = False


@click.command(help="A simple scrcpy client")
@click.option(
    "--max_width",
    default=1280,
    show_default=True,
    help="Set max width of the window",
)
@click.option(
    "--device",
    help="Select device manually (device serial required)",
)
def main(max_width: int, device: Optional[str]):
    m = MainWindow(max_width, device)
    m.show()

    m.client.start()
    while m.alive:
        m.client.start()


if __name__ == "__main__":
    main()
