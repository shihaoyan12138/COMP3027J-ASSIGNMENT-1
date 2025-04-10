from tqdm import tqdm
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader
from torchtoolbox.tools import mixup_data, mixup_criterion
from option import args_parser
from val import validate

if __name__ == '__main__':
    args = args_parser()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    torch.manual_seed(args.seed)

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.norm_mean, std=args.norm_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.norm_mean, std=args.norm_std)
    ])

    # Dataset and DataLoader
    train_dataset = datasets.ImageFolder(args.train_dir, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, pin_memory=True)

    val_dataset = datasets.ImageFolder(args.val_dir, transform=transform_val)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, pin_memory=True)

    # Load model
    if args.teacher_model == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif args.teacher_model == 'efficientnet_b0"':
        model = models.efficientnet_b0(pretrained=True)
    else:
        raise ValueError('unknown model')

    # Model initialization
    model.head = nn.Linear(model.head.in_features, args.num_classes)
    nn.init.xavier_uniform_(model.head.weight)
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-9)

    # Training loop
    criterion = nn.CrossEntropyLoss()
    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = 0

        for images, labels in tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch}'):
            images, labels = images.to(device), labels.to(device)
            inputs, targets_a, targets_b, lam = mixup_data(images, labels, args.alpha)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # val
        val_acc = validate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
        avg_loss = train_loss / len(train_loader)

        # update learning rate
        scheduler.step()

        print(f"Epoch {epoch} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Best Acc: {best_acc:.2f}%")

        # Save checkpoint
        if epoch % args.save_freq == 0 and epoch >= 30:
            print(f"\n=> Saving checkpoint at epoch {epoch}")
            ckpt_path = os.path.join(args.save_dir, f"{args.teacher_model}_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Model saved to: {ckpt_path}")
