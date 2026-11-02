import torch
import torch.nn.functional as F


def logits_update(logits, step=0, total_steps=50, temp=1):
    """
    Transition phase from raw logits to a smoothed representation.
    Linearly interpolates between the raw logits and their softmax.
    """
    coeff = (step + 1) / total_steps
    seq_soft = F.softmax(logits / temp, dim=-1)
    seq_repre = (1 - coeff) * logits + coeff * seq_soft
    return seq_repre


def soft_update(logits, step=0, total_steps=50, temp_start=1, temp_end=0.01):
    """
    Soft phase: compute a temperature-controlled softmax.
    The temperature is annealed quadratically from temp_start to temp_end over total_steps.
    """
    temp = temp_start + (temp_end - temp_start) * (1 - (step + 1) / total_steps) ** 2
    seq_soft = F.softmax(logits / temp, dim=-1)
    return seq_soft


def hard_update(logits, step=0, total_steps=50, temp_start=1, temp_end=0.01):
    """
    Hard phase: obtain a one-hot representation using a straight-through estimator.
    """
    temp = temp_start + (temp_end - temp_start) * (1 - (step + 1) / total_steps) ** 2
    seq_soft = F.softmax(logits / temp, dim=-1)
    seq_hard_ = F.one_hot(seq_soft.argmax(-1), num_classes=seq_soft.size(-1)).float()
    seq_hard = seq_hard_ - seq_soft.detach() + seq_soft
    return seq_hard


if __name__ == "__main__":
    L = 10
    num_classes = 20

    restype_tensor = torch.zeros(L, num_classes)
    restype_tensor[torch.arange(L), torch.zeros(L, dtype=torch.long)] = 1

    print("Initial restype_tensor (one-hot):")
    print(restype_tensor[1, :])

    logits = restype_tensor.clone()

    for step in range(50):
        logits = logits_update(logits, step=step, total_steps=50, temp=1)
    print("\nLogits phase result after 50 iterations:")
    print(logits[1, :])

    soft_representation = logits.clone()
    for step in range(25):
        soft_representation = soft_update(
            soft_representation, step=step, total_steps=25, temp_start=1, temp_end=0.01
        )
    print("\nSoft phase result after 25 iterations:")
    print(soft_representation[1, :])

    hard_representation = logits.clone()
    for step in range(5):
        hard_representation = hard_update(
            hard_representation, step=step, total_steps=5, temp_start=1, temp_end=0.01
        )
    print("\nHard phase result after 5 iterations:")
    print(hard_representation[1, :])
