import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.vision_transformer import deit_tiny_patch16_224, deit_base_patch16_224
from option import args_parser


if __name__ == "__main__":
    args = args_parser()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")

    # Load model
    if args.model == 'DEiT_tiny':
        model = deit_tiny_patch16_224(pretrained=False)
    elif args.model == 'DEiT_base':
        model = deit_base_patch16_224(pretrained=False)
    else:
        raise ValueError('unknown model')

    # Model initialization
    model.head = nn.Linear(model.head.in_features, args.num_classes)
    model.load_state_dict(torch.load(args.test_model))
    model = model.to(device).eval()

    # Test transform
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(args.norm_mean, args.norm_std)
    ])

    # Dataset and DataLoader
    test_dataset = datasets.ImageFolder(args.test_dir, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    # Inference
    all_predi = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs, _ = model(images)
            predi = outputs.argmax(dim=1)

            all_predi.extend(predi.cpu().numpy())
            all_targets.extend(labels.numpy())

    # Generate report
    class_names = ('Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed',
                   'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize',
                   'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet')

    print(classification_report(all_targets, all_predi, target_names=class_names))
