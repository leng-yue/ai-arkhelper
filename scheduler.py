import cv2
import time
from typing import Optional

import numpy as np

from definition import Tasks, Actions
from predict import Predictor
import scrcpy


def visualize_frame(frame, point):
    frame = frame.copy()
    cv2.circle(frame, point, 3, (0, 0, 255), -1)
    cv2.imshow("viz", frame)
    cv2.waitKey()


if __name__ == "__main__":
    predictor = Predictor()
    client = scrcpy.Client(max_width=1280, max_fps=10)
    client.start(True)

    # tasks = [Tasks.RecentBattle, Tasks.Battle, Tasks.Home]
    # tasks = [Actions.Home, Actions.Email, Actions.RecentBattle, Actions.Battle, Actions.Home]
    tasks = [Tasks.RecentBattle, Tasks.Battle, Tasks.Home]
    last_frame: Optional[np.ndarray] = None
    finish_count = 0

    while len(tasks) > 0:
        if client.last_frame is None or (client.last_frame == last_frame).all():
            continue
        last_frame = client.last_frame

        action, score, (x, y) = predictor.predict(last_frame, tasks[0].value)
        # visualize_frame(last_frame, (x, y))
        print(action, score)
        if action == Actions.Finish:
            print(f"{tasks[0]} 已完成")
            finish_count += 1
            if finish_count >= 3:
                tasks.pop(0)
            time.sleep(1)
        else:
            finish_count = 0
        if action == Actions.Touch and score > 0.1:
            print(tasks[0], action, score, (x, y))
            client.control.touch(x, y, scrcpy.ACTION_DOWN)
            client.control.touch(x, y, scrcpy.ACTION_UP)
            time.sleep(1)
