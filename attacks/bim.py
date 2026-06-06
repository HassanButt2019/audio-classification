import torch
import torchattacks


def get_bim_attack(model, epsilon, alpha, steps):
    """Create a BIM (Basic Iterative Method) attack object.

    BIM applies FGSM iteratively with a small step size alpha, clipping
    the accumulated perturbation back into the epsilon-ball after every step.
    Also known as I-FGSM (Iterative FGSM).

    Update rule per step t:
        x_0   = x_clean
        x_t+1 = clip_eps( x_t + alpha * sign( ∇_x L(x_t, y) ), x_clean, eps )

    Args:
        model:   Trained model in eval mode.
        epsilon: Maximum L∞ perturbation budget (total, across all steps).
        alpha:   Per-step perturbation size. Typically epsilon / steps or
                 a small fixed value (e.g. 0.01). Larger alpha converges
                 faster but may overshoot the epsilon ball.
        steps:   Number of iterative FGSM steps. More steps find a stronger
                 adversarial example but cost more compute.

    Returns:
        torchattacks.BIM attack object.
    """
    return torchattacks.BIM(model, eps=epsilon, alpha=alpha, steps=steps)


def evaluate_bim(model, test_loader, epsilon, alpha, steps, device):
    """Evaluate model accuracy under BIM attack.

    Args:
        model:       Trained model.
        test_loader: DataLoader for the test fold.
        epsilon:     Total L∞ perturbation budget.
        alpha:       Per-step size.
        steps:       Number of iterative steps.
        device:      cuda / mps / cpu.

    Returns:
        Adversarial accuracy (0–100).
    """
    assert not model.training, (
        "Model must be in eval mode before running BIM. Call model.eval() first."
    )

    attack = get_bim_attack(model, epsilon, alpha, steps)

    correct = 0
    total   = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        assert images.is_floating_point(), (
            f"Input tensor must be float for BIM gradients, got {images.dtype}."
        )

        adv_images = attack(images, labels)

        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = torch.max(outputs, 1)

        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total
