top1_total = 0
top5_total = 0
n_samples = 0

with torch.no_grad():
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.topk(5, 1, True, True)

        top1_preds = preds[:, 0]
        top5_correct = (preds == labels.view(-1, 1)).any(dim=1)

        top1_total += (top1_preds == labels).sum().item()
        top5_total += top5_correct.sum().item()
        n_samples += labels.size(0)

print(f"Top-1 Accuracy: {top1_total / n_samples:.4f}")
print(f"Top-5 Accuracy: {top5_total / n_samples:.4f}")