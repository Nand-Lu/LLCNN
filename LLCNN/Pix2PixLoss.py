import math
import torch


class SSIMLoss(torch.nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.register_buffer("kernel", self._cal_gaussian_kernel(11, 1.5))
        self.L = 2.0
        self.k1 = 0.0001
        self.k2 = 0.001

    @staticmethod
    def _cal_gaussian_kernel(size, sigma):
        g = torch.Tensor([math.exp(-(x - size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(size)])
        g = g / g.sum()
        window = g.reshape([-1, 1]).matmul(g.reshape([1, -1]))
        kernel = torch.reshape(window, [1, 1, size, size]).repeat(3, 1, 1, 1)
        return kernel

    def forward(self, img0, img1):
        """
        :param img0: range in (-1, 1)
        :param img1: range in (-1, 1)
        :return: SSIM loss i.e. 1 - ssim
        """
        mu0 = torch.nn.functional.conv2d(img0, self.kernel, padding=0, groups=3)
        mu1 = torch.nn.functional.conv2d(img1, self.kernel, padding=0, groups=3)
        mu0_sq = torch.pow(mu0, 2)
        mu1_sq = torch.pow(mu1, 2)
        var0 = torch.nn.functional.conv2d(img0 * img0, self.kernel, padding=0, groups=3) - mu0_sq
        var1 = torch.nn.functional.conv2d(img1 * img1, self.kernel, padding=0, groups=3) - mu1_sq
        covar = torch.nn.functional.conv2d(img0 * img1, self.kernel, padding=0, groups=3) - mu0 * mu1
        c1 = (self.k1 * self.L) ** 2
        c2 = (self.k2 * self.L) ** 2
        ssim_numerator = (2 * mu0 * mu1 + c1) * (2 * covar + c2)
        ssim_denominator = (mu0_sq + mu1_sq + c1) * (var0 + var1 + c2)
        ssim = ssim_numerator / ssim_denominator
        ssim_loss = 1.0 - ssim
        return ssim_loss


class MixedPix2PixLoss(torch.nn.Module):
    def __init__(self):
        super(MixedPix2PixLoss, self).__init__()
        self.alpha = 0.5
        self.ssim_loss = SSIMLoss()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, pred, target):
        """
        :param pred: (bs, c, h, w) image ranging in (-1, 1)
        :param target: (bs, c, h, w) image ranging in (-1, 1)
        :param reduce: (str) reduction method, "mean" or "none" or "sum"
        :return:
        """
        ssim_loss = torch.mean(self.ssim_loss(pred, target))
        l1_loss = self.l1_loss(pred, target)
        weighted_mixed_loss = self.alpha * ssim_loss + (1.0 - self.alpha) * l1_loss
        return weighted_mixed_loss