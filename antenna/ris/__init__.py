from .simulate_ris import RISSimulator
from torch.functional import F
def custom_loss(prediction, target, loss_type='SmoothL1Loss'):

    # 基本條件
    mask_20 = target == -20
    mask_b_20 = prediction[mask_20] > -20

    mask_0 = target == 0
    mask_s_0 = prediction[mask_0] < -3

    # 為了確保有梯度，設定條件不滿足時也會計入一個 dummy loss
    if mask_b_20.sum() > 0:
        loss_20 = F.smooth_l1_loss(
            prediction[mask_20][mask_b_20],
            target[mask_20][mask_b_20]
        )
    else:
        # 使用全體 prediction 的一小部分作 dummy loss，保證梯度
        loss_20 = 0.01 * F.mse_loss(prediction, target)

    if mask_s_0.sum() > 0:
        loss_0 = F.smooth_l1_loss(
            prediction[mask_0][mask_s_0],
            target[mask_0][mask_s_0]
        )
    else:
        loss_0 = 0.01 * F.mse_loss(prediction, target)

    loss = loss_20 + loss_0

    return loss