"""
    Most borrow from: https://github.com/Alibaba-MIIL/ASL
"""
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg, gamma_pos, clip, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps, max=1-self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps, max=1-self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        # loss = -loss.sum(-1)
        return -torch.mean(loss)

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                            self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)   
                self.loss *= self.asymmetric_w         
        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1) * 1000

        return _loss

class FocalDiceLoss(nn.Module):
    """Multi-label focal-dice loss"""

    def __init__(self, p_pos=2, p_neg=2, clip_pos=0.7, clip_neg=0.5, pos_weight=0.3, reduction='mean'):
        super(FocalDiceLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.pos_weight = pos_weight

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        # predict = input
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        xs_pos = predict
        p_pos = predict
        if self.clip_pos is not None and self.clip_pos >= 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)  # dim=1 按行相加
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)

        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg >= 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss = loss_pos * self.pos_weight + loss_neg * (1 - self.pos_weight)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class BCEFocalLoss(nn.Module):
    """Focal loss"""
    def __init__(self, gamma=2, reduction='mean', class_weight=None):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.reducation = reduction
        self.class_weight = class_weight

    def forward(self, data, label):
        sigmoid = nn.Sigmoid()
        pt = sigmoid(data).detach()
        if self.class_weight is not None:
            label_weight = ((1 - pt) ** self.gamma) * self.class_weight
            # label_weight = torch.exp((1 - pt)) * self.class_weight
            # label_weight = torch.exp((2 * (1 - pt)) ** self.gamma) * self.class_weight
        else:
            label_weight = (1 - pt) ** self.gamma
            # label_weight = torch.exp((1 - pt))
            # label_weight = torch.exp((2 * (1 - pt)) ** self.gamma)

        focal_loss = nn.BCEWithLogitsLoss(weight=label_weight, reduction='mean')
        return focal_loss(data, label)

class BinaryDiceLoss(nn.Module):
    """Dice loss"""
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1)

        loss = 1 - (2 * num) / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))



