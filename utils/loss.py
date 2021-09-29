import chamferdist
import torch
import torch.nn.functional as F

chamfer = chamferdist.ChamferDistance()
def chamfer_loss(flow, flow_pred, pc1, pc2):
    predicted = pc1 + flow_pred

    loss = chamfer(predicted.type(torch.float), pc2.type(torch.float), bidirectional=True) * 1e-7
    return loss


def rigidity_loss(flow, flow_pred, pc1, position1):
    source_dist1 = torch.Tensor().cuda()
    source_dist2 = torch.Tensor().cuda()
    predict_dist1 = torch.Tensor().cuda()
    predict_dist2 = torch.Tensor().cuda()
    for idx in range(pc1.shape[0]):
        for p1 in position1:
            p1 = p1.type(torch.int).cuda()

            source_dist1 = torch.cat((source_dist1, torch.index_select(pc1[idx, ...], 1, p1[idx, :])[..., None]
                                      .expand(-1, -1, p1.size()[1]).reshape(3, -1).T), dim=0)

            source_dist2 = torch.cat((source_dist2, torch.index_select(pc1[idx, ...], 1, p1[idx, :])[None, ...]
                                      .expand(p1.size()[1], -1, -1).reshape(3, -1).T), dim=0)

            predict_dist1 = torch.cat((predict_dist1, torch.index_select(pc1[idx, ...] + flow_pred[idx, ...], 1, p1[idx, :])[..., None]
                                       .expand(-1, -1, p1.size()[1]).reshape(3, -1).T), dim=0)

            predict_dist2 = torch.cat((predict_dist2, torch.index_select(pc1[idx, ...] + flow_pred[idx, ...], 1, p1[idx, :])[None, ...]
                                       .expand(p1.size()[1], -1, -1).reshape(3, -1).T), dim=0)
    loss = torch.abs(torch.sqrt(F.mse_loss(source_dist1, source_dist2)) -
                      torch.sqrt(F.mse_loss(predict_dist1, predict_dist2))) / 5
    return loss


def biomechanical_loss(constraint, flow, flow_pred, idx, pc1):
    source = pc1[idx, :, constraint[idx]]
    predicted = pc1[idx, :, constraint[idx]] + flow_pred[idx, :, constraint[idx]]
    loss = torch.tensor([0.0], device=flow.device, dtype=flow.dtype)
    for j in range(0, constraint.size(1) - 1, 2):
        loss += 1e-2 * torch.abs(torch.linalg.norm(source[:, j], source[:, j + 1]) -
                                 torch.linalg.norm(predicted[:, j], predicted[:, j + 1]))
    return loss

