import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models as m
from models.vision_transformer import deit_tiny_patch16_224, deit_base_patch16_224
from torchtoolbox.tools import mixup_data, mixup_criterion
from option import args_parser
from val import validate


if __name__ == "__main__":
    args = args_parser()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    torch.manual_seed(args.seed)

    # Data transforms
    if args.data_argu:
        print('data argumentation! super model is coming!')
        transform_train = transforms.Compose([
            # 尺寸处理
            transforms.Resize((256, 256)),  # 等比缩放保留更多细节
            transforms.RandomCrop(224),  # 随机裁剪增加位置鲁棒性

            # 几何增强
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),

            # 颜色增强
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),

            # 细节增强
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

            transforms.ToTensor(),

            # 遮挡增强
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),

            # Normalization
            transforms.Normalize(mean=args.norm_mean, std=args.norm_std)
        ])
    else:
        print('no data argumentation')
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

    # Load models
    if args.pre_train_model == 'DEiT_tiny':
        model = deit_tiny_patch16_224(pretrained=True)
    elif args.pre_train_model == 'DEiT_base':
        model = deit_base_patch16_224(pretrained=True)
    else:
        raise ValueError('unknown model')
    if args.teacher_model == 'resnet34':
        teacher_model = m.resnet34()

    elif args.teacher_model == 'efficientnet_b0':
        teacher_model = m.efficientnet_b0()
    else:
        raise ValueError('unknown teacher model')

    # Model initialization
    model.head = nn.Linear(model.head.in_features, args.num_classes)
    nn.init.xavier_uniform_(model.head.weight)
    model = model.to(device)

    teacher_model.head = nn.Linear(model.head.in_features, args.num_classes)
    teacher_model.load_state_dict(torch.load(args.teacher_path))
    teacher_model = model.to(device).eval()

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

            # Get teacher output
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            # Get student output
            outputs = model(inputs)

            # Calculate losses
            loss_teacher = criterion(outputs, teacher_outputs)
            loss_student = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # Calculate final loss
            loss = args.alpha * loss_teacher + (1 - args.alpha) * loss_student

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
            if args.data_argu:
                ckpt_path = os.path.join(args.save_dir, f"super_model_{epoch}.pth")
            else:
                ckpt_path = os.path.join(args.save_dir, f"model_{epoch}.pth")

            torch.save(model.state_dict(), ckpt_path)
            print(f"Model saved to: {ckpt_path}")
