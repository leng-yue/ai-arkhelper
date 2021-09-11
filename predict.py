import numpy as np
import torch
from torch.nn.functional import avg_pool2d

from definition import ACTIONS
from model.net import ArkNet
import torchvision.transforms as T


class Predictor:
    def __init__(self):
        self.model = ArkNet()
        self.model.load_state_dict(torch.load("checkpoints/best.model"))
        self.model.eval()
        self.model.cuda()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image, action):
        action_idx = ACTIONS.index(action)
        raw_h, raw_w = image.shape[:2]
        image = self.transform(image)  # 1, H, W
        new_h, new_w = image.shape[1:]

        # OneHot 编码操作空间
        actions = np.zeros((len(ACTIONS), new_h, new_w))
        actions[action_idx] = 1
        image = np.concatenate([image, actions]).astype(np.float32)

        # 预测
        images = torch.FloatTensor(image).unsqueeze(0).cuda()
        predict_finished, predict_hm, predict_regs_wh = self.model(images)
        predict_finished = predict_finished.argmax(1)
        # predict_hm = avg_pool2d(predict_hm, 3, 1, 1)

        argmax = predict_hm.flatten().argmax()
        row = argmax // predict_hm.shape[3]
        col = argmax % predict_hm.shape[3]
        score = predict_hm[0, 0, row, col]

        return bool(predict_finished[0] == 1), float(score), (
            int((col / predict_hm.shape[3]) * raw_w),
            int((row / predict_hm.shape[2]) * raw_h)
        )
