model.eval()
num_shown = 0

for i in range(len(patch_adv_dataset)):
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