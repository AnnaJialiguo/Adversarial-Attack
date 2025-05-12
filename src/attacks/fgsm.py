import torch.nn.functional as F

def fgsm_attack(image, label, model, epsilon):
    image.requires_grad = True
    output = model(image)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    adv_image = image + epsilon * image.grad.data.sign()
    adv_image = torch.clamp(adv_image, 0, 1)
    return adv_image.detach()

adv_images = []
adv_labels = []

model.eval()
epsilon = 0.02

for images, labels in tqdm(loader):
    images, labels = images.to(device), labels.to(device)
    for i in range(images.size(0)):
        img = images[i].unsqueeze(0)
        lbl = labels[i].unsqueeze(0).to(device)
        adv = fgsm_attack(img, lbl, model, epsilon)
        adv_images.append(adv.squeeze(0).cpu())
        adv_labels.append(lbl.squeeze(0).cpu())

top1_total = 0
top5_total = 0
n_samples = 0

model.eval()

with torch.no_grad():
    for images, labels in tqdm(adv_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.topk(5, 1, True, True)

        top1_preds = preds[:, 0]
        top5_correct = (preds == labels.view(-1, 1)).any(dim=1)

        top1_total += (top1_preds == labels).sum().item()
        top5_total += top5_correct.sum().item()
        n_samples += labels.size(0)
        
if n_samples > 0:
    print(f"FGSM Attack Top-1 Accuracy: {top1_total / n_samples:.4f}")
    print(f"FGSM Attack Top-5 Accuracy: {top5_total / n_samples:.4f}")
else:
    print("n_samples is 0 — check your DataLoader or attack generation.")

print("Transfer to DenseNet-121")
acc_results = {}

acc_results["Original"] = evaluate_model(transfer_model, loader)
acc_results["FGSM"] = evaluate_model(transfer_model, adv_loader)
acc_results["PGD"] = evaluate_model(transfer_model, pgd_adv_loader)
acc_results["Patch"] = evaluate_model(transfer_model, patch_adv_loader)

for name, (top1, top5) in acc_results.items():
    print(f"{name} → Top-1: {top1:.4f}, Top-5: {top5:.4f}")