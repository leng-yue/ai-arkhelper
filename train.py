import torch
import torchvision.transforms as T
from torch.nn import SmoothL1Loss, CrossEntropyLoss
from tqdm import tqdm
import imgaug.augmenters as iaa
from torch.utils.data import DataLoader

from definition import Actions, ACTIONS
from model.dataset import ArkDataset
from model.miou import centernet_to_standard, mean_iou
from model.net import ArkNet
from model.loss import RegL1Loss, FocalLoss


def train():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_aug = iaa.Sequential([
        iaa.OneOf([
            iaa.GaussianBlur((0, 2.0)),
            iaa.AverageBlur((1, 3)),
            iaa.MedianBlur((1, 3)),
        ]),
        iaa.MultiplyBrightness(mul=(0.65, 1.35)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    ]).augment_image

    train_set = ArkDataset(
        "records",
        transform=T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    valid_set = ArkDataset(
        "records",
        transform=T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    # import cv2
    # for i in valid_set:
    #     image = i[0][:3].transpose(1, 2, 0) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    #     image = cv2.resize(image, (1280, 720))
    #     cv2.imshow("viz", image)
    #     cv2.waitKey(1000)
    # exit()

    train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True, num_workers=8)
    valid_loader = DataLoader(dataset=valid_set, batch_size=4, shuffle=True, num_workers=4)

    model = ArkNet()
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='min')
    reg_loss = RegL1Loss()
    focal_loss = FocalLoss()
    cross_entropy_loss = CrossEntropyLoss()
    min_loss = 9999

    for epoch in range(0, 1000):
        # Train
        model.train()
        bar = tqdm(train_loader, 'Training', ascii=True)
        losses = []

        for image, result, hm, regs_wh, ind_masks in bar:
            image, result, hm = image.to(DEVICE), result.to(DEVICE), hm.to(DEVICE)
            regs_wh, ind_masks = regs_wh.to(DEVICE), ind_masks.to(DEVICE)

            # 预测
            predict_result, predict_hm, predict_regs_wh = model(image)

            # 提取需要关注点的部分
            focus_index = result == ACTIONS.index(Actions.Touch.value)
            hm, regs_wh, ind_masks = hm[focus_index], regs_wh[focus_index], ind_masks[focus_index]
            predict_hm, predict_regs_wh = predict_hm[focus_index], predict_regs_wh[focus_index]

            # 进行分类损失计算
            predict_result = predict_result.squeeze(1)
            loss = cross_entropy_loss(predict_result, result)

            # 如果需要计算 HM
            if predict_hm.shape[0] > 0:
                predict_center, predict_bias = torch.split(predict_regs_wh, 2, 1)
                center, bias = torch.split(regs_wh, 2, 1)
                ind_masks_center, ind_masks_bias = torch.split(ind_masks, 2, 1)

                center_loss = reg_loss(predict_center, center, ind_masks_center)
                bias_loss = reg_loss(predict_bias, bias, ind_masks_bias)
                heatmap_loss = focal_loss(predict_hm, hm)

                # 加权计算
                loss += center_loss * 0.1 + bias_loss + heatmap_loss

            losses.append(float(loss))

            # 计算状态正确率
            action_acc = (predict_result.argmax(1) == result).sum() / len(result)

            # 快乐三步曲
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]['lr']
            bar.set_description("Train epoch %d, loss %.4f, avg loss %.4f, Action Acc %.4f, lr %.6f" % (
                epoch, float(loss), sum(losses) / len(losses), action_acc, lr
            ))

            # predict_hm_slice = predict_hm[0].cpu().squeeze().detach().numpy()
            # print(predict_hm_slice.shape)
            # cv2.imshow("HeatMap", predict_hm_slice)
            # cv2.waitKey(10)

        # Valid
        # model.eval()
        # bar = tqdm(valid_loader, 'Validating', ascii=True)
        # losses = []
        #
        # for image, result, hm, regs_wh, ind_masks in bar:
        #     image, result, hm = image.to(DEVICE), result.to(DEVICE), hm.to(DEVICE)
        #     regs_wh, ind_masks = regs_wh.to(DEVICE), ind_masks.to(DEVICE)
        #     # 预测
        #     predict_result, predict_hm, predict_regs_wh = model(image)
        #
        #     # 提取需要关注点的部分
        #     focus_index = result == ACTIONS.index(Actions.Touch.value)
        #     hm, regs_wh, ind_masks = hm[focus_index], regs_wh[focus_index], ind_masks[focus_index]
        #     predict_hm, predict_regs_wh = predict_hm[focus_index], predict_regs_wh[focus_index]
        #
        #     # 进行分类损失计算
        #     predict_result = predict_result.squeeze(1)
        #     loss = cross_entropy_loss(predict_result, result)
        #
        #     # 如果需要计算 HM
        #     if predict_hm.shape[0] > 0:
        #         predict_center, predict_bias = torch.split(predict_regs_wh, 2, 1)
        #         center, bias = torch.split(regs_wh, 2, 1)
        #         ind_masks_center, ind_masks_bias = torch.split(ind_masks, 2, 1)
        #
        #         center_loss = reg_loss(predict_center, center, ind_masks_center)
        #         bias_loss = reg_loss(predict_bias, bias, ind_masks_bias)
        #         heatmap_loss = focal_loss(predict_hm, hm)
        #
        #         # 加权计算
        #         loss += center_loss * 0.1 + bias_loss + heatmap_loss
        #
        #     losses.append(float(loss))
        #
        #     # 计算状态正确率
        #     action_acc = (predict_result.argmax(1) == result).sum() / len(result)
        #     lr = optimizer.param_groups[0]['lr']
        #     bar.set_description("Valid epoch %d, loss %.4f, avg loss %.4f, Action Acc %.4f, lr %.6f" % (
        #         epoch, float(loss), sum(losses) / len(losses), action_acc, lr
        #     ))
        #
        #     # predict_hm_slice = predict_hm[0].cpu().squeeze().detach().numpy()
        #     # cv2.imshow("HeatMap", predict_hm_slice)
        #     # cv2.waitKey(10)

        avg_valid_loss = sum(losses) / len(losses)
        scheduler.step(avg_valid_loss)
        if avg_valid_loss < min_loss:
            min_loss = avg_valid_loss
            torch.save(model.state_dict(), "checkpoints/best.model")
            print('Model Saved')
        # torch.save(model.state_dict(), "models/save_%d.model" % epoch)


if __name__ == "__main__":
    train()
