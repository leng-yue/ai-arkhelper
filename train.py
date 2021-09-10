import torch
import torchvision.transforms as T
from torch.nn import SmoothL1Loss
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.dataset import ArkDataset
from model.miou import centernet_to_standard, mean_iou
from model.net import ArkNet
from model.loss import RegL1Loss, FocalLoss


def train():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = ArkDataset(
        "records",
        transform=T.Compose([
            T.ToPILImage(),
            # T.ColorJitter(brightness=.15, contrast=.15),
            T.Resize((512, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    valid_set = ArkDataset(
        "records",
        transform=T.Compose([
            T.ToPILImage(),
            T.Resize((512, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    # import cv2
    # data = train_set.__getitem__(2)
    # # .permute(1, 2, 0).detach().numpy()
    # cv2.imshow("128x128", data['hm'].transpose(1, 2, 0))
    # cv2.waitKey()
    # exit()

    train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_set, batch_size=4, shuffle=False, num_workers=1)

    # label_map = train_set.get_label_map()
    # print(len(label_map), label_map)

    model = ArkNet()
    # model.load_state_dict(torch.load('models/best.model'))
    # for param in model.backbone.parameters():
    #     param.requires_grad = False

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='min')
    reg_loss = RegL1Loss()
    focal_loss = FocalLoss()
    l1_loss = SmoothL1Loss()
    min_loss = 9999

    for epoch in range(0, 1000):
        # Train
        model.train()
        bar = tqdm(train_loader, 'Training', ascii=True)
        losses = []

        for image, finished, hm, regs_wh, ind_masks in bar:
            image, finished, hm = image.to(DEVICE), finished.type(torch.float32).to(DEVICE), hm.to(DEVICE)
            regs_wh, ind_masks = regs_wh.to(DEVICE), ind_masks.to(DEVICE)

            predict_finished, predict_hm, predict_regs_wh = model(image)
            predict_finished = predict_finished.squeeze(1)
            predict_center, predict_bias = torch.split(predict_regs_wh, 2, 1)
            center, bias = torch.split(regs_wh, 2, 1)
            ind_masks_center, ind_masks_bias = torch.split(ind_masks, 2, 1)

            center_loss = reg_loss(predict_center, center, ind_masks_center)
            bias_loss = reg_loss(predict_bias, bias, ind_masks_bias)
            heatmap_loss = focal_loss(predict_hm, hm)
            finish_loss = l1_loss(predict_finished, finished)

            # 加权计算
            loss = center_loss * 0.1 + bias_loss + heatmap_loss + finish_loss
            losses.append(float(loss))

            # 快乐三步曲
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]['lr']
            bar.set_description("Train epoch %d, loss %.4f, avg loss %.4f, lr %.6f" % (
                epoch, float(loss), sum(losses) / len(losses), lr
            ))

            # predict_hm_slice = predict_hm[0].cpu().squeeze().detach().numpy()
            # print(predict_hm_slice.shape)
            # cv2.imshow("HeatMap", predict_hm_slice)
            # cv2.waitKey(10)

        # Valid
        model.eval()
        bar = tqdm(valid_loader, 'Validating', ascii=True)
        losses = []

        for image, finished, hm, regs_wh, ind_masks in bar:
            image, finished, hm = image.to(DEVICE), finished.to(DEVICE), hm.to(DEVICE)
            regs_wh, ind_masks = regs_wh.to(DEVICE), ind_masks.to(DEVICE)

            predict_finished, predict_hm, predict_regs_wh = model(image)
            predict_finished = predict_finished.squeeze(1)
            predict_center, predict_bias = torch.split(predict_regs_wh, 2, 1)
            center, bias = torch.split(regs_wh, 2, 1)
            ind_masks_center, ind_masks_bias = torch.split(ind_masks, 2, 1)

            center_loss = reg_loss(predict_center, center, ind_masks_center)
            bias_loss = reg_loss(predict_bias, bias, ind_masks_bias)
            heatmap_loss = focal_loss(predict_hm, hm)
            finish_loss = l1_loss(predict_finished, finished)

            # 加权计算
            loss = center_loss * 0.1 + bias_loss + heatmap_loss + finish_loss
            losses.append(float(loss))

            # 计算状态正确率
            finished_acc = (predict_finished.round() == finished).sum() / len(finished)

            lr = optimizer.param_groups[0]['lr']
            bar.set_description("Valid epoch %d, loss %.4f, avg loss %.4f, Finished Acc %.4f, lr %.6f" % (
                epoch, float(loss), sum(losses) / len(losses), finished_acc, lr
            ))

            # predict_hm_slice = predict_hm[0].cpu().squeeze().detach().numpy()
            # cv2.imshow("HeatMap", predict_hm_slice)
            # cv2.waitKey(10)

        avg_valid_loss = sum(losses) / len(losses)
        scheduler.step(avg_valid_loss)
        if avg_valid_loss < min_loss:
            min_loss = avg_valid_loss
            torch.save(model.state_dict(), "checkpoints/best.model")
            print('Model Saved')
        # torch.save(model.state_dict(), "models/save_%d.model" % epoch)


if __name__ == "__main__":
    train()
