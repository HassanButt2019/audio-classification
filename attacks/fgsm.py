
import torch
import torchattacks

def get_fgsm_attack(model, epsilon):
    """
    Create FGSM attack object
    
    model:   your trained CNN model
    epsilon: perturbation strength
    
    returns: attack object
    """
    attack = torchattacks.FGSM(model, eps=epsilon)
    return attack


def evaluate_fgsm(model, test_loader, epsilon, device):
    """
    Evaluate model accuracy under FGSM attack

    model:       trained CNN model
    test_loader: DataLoader for test fold
    epsilon:     perturbation strength
    device:      cuda or cpu

    returns: adversarial accuracy (0–100)
    """

    # Check 1 — model must be in eval mode so dropout is disabled and
    # batch-norm uses running statistics, giving deterministic predictions.
    assert not model.training, (
        "Model must be in eval mode before running FGSM. Call model.eval() first."
    )

    attack = get_fgsm_attack(model, epsilon)

    correct = 0
    total = 0

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Check 2 — FGSM needs ∇_x Loss. torchattacks enables requires_grad
        # internally, but gradients are undefined on integer tensors. Assert
        # float dtype here to catch any accidental cast or wrong DataLoader transform.
        assert images.is_floating_point(), (
            f"Input tensor must be float for FGSM gradients, got {images.dtype}. "
            "Check your DataLoader / preprocessing pipeline."
        )

        adv_images = attack(images, labels)

        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    adversarial_accuracy = 100 * correct / total
    return adversarial_accuracy