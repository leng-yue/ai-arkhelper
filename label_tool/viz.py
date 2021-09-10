from pathlib import Path

import cv2
import numpy as np
from matplotlib import font_manager
import matplotlib.pyplot as plt

fontP = font_manager.FontProperties()
fontP.set_family('SimHei')
fontP.set_size(14)

plt.ion()
plt.rcParams["figure.figsize"] = (12, 7)
for i in Path("records").rglob("**/*.jpg"):
    temp = i.name.replace(".jpg", "").split("_")[1:]
    plt.title(
        f'类型: {i.parent.name}, 状态: {temp[0]}',
        fontproperties=fontP
    )
    im = np.fromfile(str(i.absolute()), dtype=np.uint8)
    image = cv2.imdecode(im, cv2.IMREAD_COLOR)
    for box in temp[1:]:
        x0, y0, x1, y1 = list(map(int, box.split(",")))
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 3)
        cv2.circle(image, ((x0 + x1) // 2, (y0 + y1) // 2), 5, (0, 0, 255), -1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    # plt.show()
    plt.waitforbuttonpress()
