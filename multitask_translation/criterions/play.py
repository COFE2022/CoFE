import torch
import torch.nn.functional as F


def loss_label_smoothing_weighted(log_prob, label, ignore_index=None, label_smoothing: float = 0, weights=None):
    if weights is None:
        weights = torch.ones(label.shape[0], device=label.device).view(-1, 1)
    else:
        weights = weights.view(-1, 1)
    label = label.clone().detach()
    if label.dim() == log_prob.dim() - 1:
        label = label.unsqueeze(-1)

    if ignore_index is not None:
        padding_mask = label.eq(ignore_index)
        label.clamp_min_(0)  # [b,max_len, 1]
        nll_loss = -log_prob.gather(dim=-1, index=label)
        # works for fp16 input tensor too, by internally upcasting it to fp32

        smoothed_loss = -log_prob.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)
        nll_loss = nll_loss.squeeze(-1)
        smoothed_loss = smoothed_loss.squeeze(-1)
        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        active_elements = torch.ones_like(nll_loss) - padding_mask.squeeze(-1).long()
        active_elements = active_elements * weights
        weighted_nll_loss = nll_loss * weights
        weighted_nll_loss = weighted_nll_loss.sum() / active_elements.sum()
        smoothed_loss = smoothed_loss.sum() / (active_elements.sum() * log_prob.shape[-1])
        total_loss = (1 - label_smoothing) * weighted_nll_loss + label_smoothing * smoothed_loss
        nll_loss = nll_loss.sum() / (torch.ones_like(nll_loss) - padding_mask.squeeze(-1).long()).sum()
        return total_loss, nll_loss
    else:
        nll_loss = -log_prob.gather(dim=-1, index=label)
        smoothed_loss = -log_prob.sum(dim=-1, keepdim=True, dtype=torch.float32)
        nll_loss = nll_loss.squeeze(-1)
        smoothed_loss = smoothed_loss.squeeze(-1)
        active_elements = torch.ones_like(nll_loss)
        weighted_nll_loss = nll_loss * weights
        weighted_nll_loss = weighted_nll_loss.sum() / active_elements.sum()
        smoothed_loss = smoothed_loss.sum() / (active_elements.sum() * log_prob.shape[-1])
        total_loss = weighted_nll_loss * (1.0 - label_smoothing) + label_smoothing * smoothed_loss
        nll_loss = nll_loss.mean()
        return total_loss, nll_loss


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss




logits = torch.randn([16, 251, 200])
log_probs = torch.log_softmax(logits, dim=-1)
labels = torch.randint(low=0, high=199, size=[16, 251])

print(label_smoothed_nll_loss(log_probs, labels, 0,reduce=True))
print(label_smoothed_nll_loss(log_probs, labels, 0,reduce=True))
print(label_smoothed_nll_loss(log_probs, labels, 0,reduce=True))
print(loss_label_smoothing_weighted(log_probs, labels, label_smoothing=0))
print(label_smoothed_nll_loss(log_probs, labels, 0,reduce=True))
print(torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), label_smoothing=0,
                                        reduction='sum'))
print("##################ignore##########################")
labels[labels == 0] = 30
print(loss_label_smoothing_weighted(log_probs, labels, label_smoothing=0.1, ignore_index=30))
print(label_smoothed_nll_loss(log_probs, labels, 0.1, ignore_index=30))
print(F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), label_smoothing=0.1, ignore_index=30,
                      reduction='sum'),
      F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), label_smoothing=0, reduction='sum',
                      ignore_index=30))

weights = torch.tensor(list(range(16)))
print(loss_label_smoothing_weighted(log_probs, labels, label_smoothing=0.1, ignore_index=30, weights=weights))
print(label_smoothed_nll_loss(log_probs.view(-1,log_probs.size(-1)), labels.view(-1), 0.1, ignore_index=30))
print(label_smoothed_nll_loss(log_probs, labels, 0.1, ignore_index=30,reduce=False)[0].sum())
print(label_smoothed_nll_loss(log_probs, labels, 0.1, ignore_index=30,reduce=False)[1].sum())