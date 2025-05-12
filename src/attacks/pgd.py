def pgd_attack(model, images, labels, epsilon=0.02, alpha=0.005, iters=10):
    ori_images = images.clone().detach()
    adv_images = ori_images.clone().detach()

    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        adv_images = adv_images + alpha * adv_images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(ori_images + eta, 0, 1).detach()

    return adv_images

pgd_adv_images = []
pgd_adv_labels = []

model.eval()
epsilon = 0.02
alpha = 0.005
iters = 10

for images, labels in tqdm(loader):
    images, labels = images.to(device), labels.to(device)
    adv = pgd_attack(model, images, labels, epsilon=epsilon, alpha=alpha, iters=iters)
    pgd_adv_images.extend(adv.cpu())
    pgd_adv_labels.extend(labels.cpu())

top1_total = 0
top5_total = 0
n_samples = 0

with torch.no_grad():
    for images, labels in tqdm(pgd_adv_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.topk(5, 1, True, True)

        top1_preds = preds[:, 0]
        top5_correct = (preds == labels.view(-1, 1)).any(dim=1)

        top1_total += (top1_preds == labels).sum().item()
        top5_total += top5_correct.sum().item()
        n_samples += labels.size(0)

print(f"PGD Attack Top-1 Accuracy: {top1_total / n_samples:.4f}")
print(f"PGD Attack Top-5 Accuracy: {top5_total / n_samples:.4f}")

model.eval()
num_shown = 0

for i in range(len(pgd_adv_dataset)):
    adv_img, true_label = adv_dataset[i]
    input_tensor = adv_img.unsqueeze(0).to(device)
    true_label = true_label.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_label = output.argmax(dim=1).item()

    if pred_label != true_label.item():
        show_comparison(loader.dataset[i][0], adv_img, idx=i)
        num_shown += 1

    if num_shown >= 3:
        break

def patch_pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, iters=10, patch_size=32):
    adv_images = images.clone().detach()
    B, C, H, W = images.shape

    for i in range(B):
        x = torch.randint(0, W - patch_size, (1,)).item()
        y = torch.randint(0, H - patch_size, (1,)).item()

        patch = adv_images[i, :, y:y+patch_size, x:x+patch_size].clone().detach()

        for _ in range(iters):
            patch.requires_grad = True
            temp_image = adv_images[i].clone().detach()
            temp_image[:, y:y+patch_size, x:x+patch_size] = patch

            output = model(temp_image.unsqueeze(0))
            loss = F.cross_entropy(output, labels[i].unsqueeze(0))
            model.zero_grad()
            loss.backward()

            patch = patch + alpha * patch.grad.sign()
            patch = torch.clamp(patch, images[i, :, y:y+patch_size, x:x+patch_size] - epsilon,
                                         images[i, :, y:y+patch_size, x:x+patch_size] + epsilon)
            patch = torch.clamp(patch, 0, 1).detach()

        adv_images[i, :, y:y+patch_size, x:x+patch_size] = patch

    return adv_images

patch_adv_images = []
patch_adv_labels = []

epsilon = 0.5
alpha = 0.01
iters = 10
patch_size = 32

for images, labels in tqdm(loader):
    images, labels = images.to(device), labels.to(device)
    adv = patch_pgd_attack(model, images, labels, epsilon=epsilon, alpha=alpha, iters=iters, patch_size=patch_size)
    patch_adv_images.extend(adv.cpu())
    patch_adv_labels.extend(labels.cpu())