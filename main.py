import torch
from torchvision import models, transforms
from src.dataloader import CustomImageNetDataset
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.attacks.patch_attack import patch_attack
from src.utils import evaluate_model

def main():
    print("Loading model...")
    model = models.resnet34(weights='IMAGENET1K_V1')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Loading dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = CustomImageNetDataset(root='./TestDataSet', transform=transform)
    print("Dataset loaded.")

    print("Evaluating model on clean data...")
    evaluate_model(model, dataset, device)

    # You can call attacks like below:
    # fgsm_attack(model, ...)
    # pgd_attack(model, ...)
    # patch_attack(model, ...)
    print("Setup complete. Run attacks separately with parameters.")

if __name__ == "__main__":
    main()

>>>>>>> a98f8ad (Upload2 project files for adversarial attack)
# FGSM result
adv_dataset1 = fgsm_attack(model, dataset, epsilon=0.02, device=device)
evaluate_model(model, adv_dataset1, device)

# PGD result
adv_dataset2 = pgd_attack(model, dataset, epsilon=0.02, alpha=0.005, steps=10, device=device)
evaluate_model(model, adv_dataset2, device)

# Patch result
adv_dataset3 = patch_attack(model, dataset, patch_size=32, epsilon=0.3, device=device)
evaluate_model(model, adv_dataset3, device)
