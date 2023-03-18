import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F


def compute_gradient(img):
    gradx = img[..., 1:, :] - img[..., :-1, :]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady


class SmoothingTVLoss(torch.nn.Module):
    def __init__(self):
        super(SmoothingTVLoss, self).__init__()
        self.eps = 1e-3

    def forward(self, data, aux, ref):
        """
        data : input feature map
        aux: input original image
        ref : label image
        """
        N, C, H, W = data.shape

        data_dw = F.pad(torch.abs(data[:, :, :, :-1] - data[:, :, :, 1:]), (1, 0, 0, 0))
        data_dh = F.pad(torch.abs(data[:, :, :-1, :] - data[:, :, 1:, :]), (0, 0, 1, 0))
        aux_dw = F.pad(torch.abs(aux[:, :, :, :-1] - aux[:, :, :, 1:]), (1, 0, 0, 0))
        aux_dh = F.pad(torch.abs(aux[:, :, :-1, :] - aux[:, :, 1:, :]), (0, 0, 1, 0))
        ref_dw = F.pad(torch.abs(ref[:, :, :, :-1] - ref[:, :, :, 1:]), (1, 0, 0, 0))
        ref_dh = F.pad(torch.abs(ref[:, :, :-1, :] - ref[:, :, 1:, :]), (0, 0, 1, 0))


        data_d = data_dw + data_dh
        ref_dw[ref_dw > 0.001] = 1
        ref_dw[ref_dw < 0.001] = 0
        ref_dh[ref_dh < 0.001] = 0
        ref_dh[ref_dh > 0.001] = 1
        ref_dw, _ = torch.max(ref_dw, dim=1, keepdim=True)
        ref_dh, _ = torch.max(ref_dh, dim=1, keepdim=True)
        aux_d, _ = torch.max(torch.clamp(aux_dw + ref_dw, 0, 1.5) + torch.clamp(aux_dh + ref_dh, 0, 1.5), dim=1)
        
        loss2 = torch.norm(data_d / (aux_d + self.eps), p=1) / (C * H * W)
        return loss2


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_gradient(predict)
        target_gradx, target_grady = compute_gradient(target)

        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


class L1GradLoss(nn.Module):
    def __init__(self):
        super(L1GradLoss, self).__init__()
        self.restor_loss = nn.L1Loss()
        self.grad_loss = GradientLoss()

    def forward(self, predict, target):
        return self.restor_loss(predict, target) + self.grad_loss(predict, target)


class GradientLoss2(nn.Module):
    def __init__(self):
        super(GradientLoss2, self).__init__()
        self.eps = 1e-2
        self.criterion = nn.MSELoss()

    def forward(self, data, aux):
        data_dw = F.pad(torch.abs(data[:, :, :, :-1] - data[:, :, :, 1:]), (1, 0, 0, 0))
        data_dh = F.pad(torch.abs(data[:, :, :-1, :] - data[:, :, 1:, :]), (0, 0, 1, 0))
        aux_dw = F.pad(torch.abs(aux[:, :, :, :-1] - aux[:, :, :, 1:]), (1, 0, 0, 0))
        aux_dh = F.pad(torch.abs(aux[:, :, :-1, :] - aux[:, :, 1:, :]), (0, 0, 1, 0))

        data_d = data_dw + data_dh
        aux_d = aux_dw + aux_dh

        loss = self.criterion(data_d, aux_d)
        return loss


class PerLoss(torch.nn.Module):
    def __init__(self):
        super(PerLoss, self).__init__()
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.to('cuda')
        for param in vgg_model.parameters():
            param.requires_grad = False

        self.vgg_layers = vgg_model

        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, data, gt):
        loss = []
        if data.shape[1] == 1:
            data = data.repeat(1, 3, 1, 1)
            gt = gt.repeat(1, 3, 1, 1)

        dehaze_features = self.output_features(data)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss) / len(loss)


class PerL1Loss(torch.nn.Module):
    def __init__(self):
        super(PerL1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.per_loss_func = PerLoss().to('cuda')

    def forward(self, output, target):
        l1_loss = self.l1_loss_func(output, target)
        per_loss = self.per_loss_func(output, target)
        # total_loss = l1_loss + 0.04 * per_loss
        total_loss = l1_loss + 0.2 * per_loss
        return total_loss

class TVLoss(torch.nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, data):
        w_variance = torch.sum(torch.pow(data[:, :, :, :-1] - data[:, :, :, 1:], 2))
        h_variance = torch.sum(torch.pow(data[:, :, :-1, :] - data[:, :, 1:, :], 2))

        count_h = self._tensor_size(data[:, :, 1:, :])
        count_w = self._tensor_size(data[:, :, :, 1:])

        tv_loss = h_variance / count_h + w_variance / count_w
        return tv_loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def safe_div(a, b, eps=1e-2):
    return a / torch.clamp_min(b, eps)


class WTVLoss(torch.nn.Module):
    def __init__(self):
        super(WTVLoss, self).__init__()
        self.eps = 1e-2

    def forward(self, data, aux):
        data_dw = data[:, :, :, :-1] - data[:, :, :, 1:]
        data_dh = data[:, :, :-1, :] - data[:, :, 1:, :]
        aux_dw = torch.abs(aux[:, :, :, :-1] - aux[:, :, :, 1:])
        aux_dh = torch.abs(aux[:, :, :-1, :] - aux[:, :, 1:, :])

        w_variance = torch.sum(torch.pow(safe_div(data_dw, aux_dw, self.eps), 2))
        h_variance = torch.sum(torch.pow(safe_div(data_dh, aux_dh, self.eps), 2))

        count_h = self._tensor_size(data[:, :, 1:, :])
        count_w = self._tensor_size(data[:, :, :, 1:])

        tv_loss = h_variance / count_h + w_variance / count_w
        return tv_loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class WTVLoss2(torch.nn.Module):
    def __init__(self):
        super(WTVLoss2, self).__init__()
        self.eps = 1e-2
        self.criterion = nn.MSELoss()

    def forward(self, data, aux):
        N, C, H, W = data.shape

        data_dw = F.pad(torch.abs(data[:, :, :, :-1] - data[:, :, :, 1:]), (1, 0, 0, 0))
        data_dh = F.pad(torch.abs(data[:, :, :-1, :] - data[:, :, 1:, :]), (0, 0, 1, 0))
        aux_dw = F.pad(torch.abs(aux[:, :, :, :-1] - aux[:, :, :, 1:]), (1, 0, 0, 0))
        aux_dh = F.pad(torch.abs(aux[:, :, :-1, :] - aux[:, :, 1:, :]), (0, 0, 1, 0))

        data_d = data_dw + data_dh
        aux_d = aux_dw + aux_dh

        loss1 = self.criterion(data_d, aux_d)
        # loss2 = torch.norm(data_d / (aux_d + self.eps), p=1) / (C * H * W)
        loss2 = torch.norm(data_d / (aux_d + self.eps)) / (C * H * W)
        return loss1 + loss2 * 4.0


class WTVLoss3(torch.nn.Module):
    def __init__(self):
        super(WTVLoss3, self).__init__()
        self.eps = 1e-2
        self.criterion = nn.L1Loss()

    def forward(self, data, aux):
        N, C, H, W = data.shape

        data_dw = F.pad(torch.abs(data[:, :, :, :-1] - data[:, :, :, 1:]), (1, 0, 0, 0))
        data_dh = F.pad(torch.abs(data[:, :, :-1, :] - data[:, :, 1:, :]), (0, 0, 1, 0))
        aux_dw = F.pad(torch.abs(aux[:, :, :, :-1] - aux[:, :, :, 1:]), (1, 0, 0, 0))
        aux_dh = F.pad(torch.abs(aux[:, :, :-1, :] - aux[:, :, 1:, :]), (0, 0, 1, 0))

        data_d = data_dw + data_dh
        aux_d = aux_dw + aux_dh

        loss1 = self.criterion(data_d, aux_d)
        # loss2 = torch.norm(data_d / (aux_d + self.eps), p=1) / (C * H * W)
        loss2 = torch.norm(data_d / (aux_d + self.eps)) / (C * H * W)
        return loss1 + loss2 * 4.0



if __name__ == "__main__":
    pass
