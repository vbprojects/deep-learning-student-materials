from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import urllib.request
from tqdm import tqdm

import time
from torchvision import models
import random


class TransformedSubset(Dataset):
    def __init__(self, base: datasets.ImageFolder, indices, transform = None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_idx = self.indices[i]
        path, target = self.base.samples[base_idx]
        img = self.base.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.base.target_transform is not None:
            target = self.base.target_transform(target)
        
        return img, target
    

def make_transforms(image_size, train_aug = "resize_flip"):
    if isinstance(image_size, int):
        size_hw = (image_size, image_size)
    else:
        size_hw = image_size
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    if train_aug == "random_resized_crop":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(size_hw, scale = (0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize(size_hw),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    val_tf = transforms.Compose([
        transforms.Resize(size_hw),
        transforms.ToTensor(),
        normalize
    ])
    return train_tf, val_tf


def set_up(seed: int, data_dir: str):
    set_seeds_to(seed)
    download_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Accelerator {device} will be used.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    base_ds = datasets.ImageFolder(data_dir)
    g = torch.Generator().manual_seed(seed)
    n = len(base_ds)
    train_size = int(0.8 * n)
    val_size = n - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        base_ds, [train_size, val_size], generator = g
    )
    train_idx = train_subset.indices
    val_idx = val_subset.indices
    return device, base_ds, train_idx, val_idx


def make_loaders(base_ds, train_idx, val_idx, seed: int, batch_size: int, num_workers: int, image_size, train_aug = "resize_flip"):
    train_tf, val_tf = make_transforms(image_size = image_size, train_aug = train_aug)
    train_ds = TransformedSubset(base_ds, train_idx, transform = train_tf)
    val_ds = TransformedSubset(base_ds, val_idx, transform = val_tf)
    g = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size = batch_size, shuffle = True, num_workers = num_workers, generator = g, pin_memory = pin
    )
    val_loader = DataLoader(
        val_ds, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = pin
    )
    return train_loader, val_loader


def download_dataset():
    url = (
        "https://firebasestorage.googleapis.com/v0/b/uva-landmark-images.appspot.com/o/"
        "dataset.zip?alt=media&token=e1403951-30d6-42b8-ba4e-394af1a2ddb7"
    )
    if not os.path.exists('dataset'):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, 'dataset.zip')
        print("Extracting dataset...")
        with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove('dataset.zip')
    else:
        print("Dataset already exists.")


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def set_seeds_to(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def plot_training_history(history, title="Training History"):
    """Plot training and validation loss/accuracy."""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"training_history_for_{title}.png")
    plt.show()


def _set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()


def train_epoch(model, dataloader, criterion, optimizer, device, feature_extract = False):
    """Train the model for one epoch."""
    model.train()
    if feature_extract:
        model.apply(_set_bn_eval)
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train_model(device, model, train_loader, val_loader, num_epochs=10, lr=0.001, feature_extract = False, step_size = 5, gamma = 0.1):
    """
    Train and evaluate a model.

    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs + 3):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, feature_extract = feature_extract
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Train Loss: {train_loss}, Train Acc: {train_acc}%')
        print(f'Val Loss: {val_loss}, Val Acc: {val_acc}%')

        scheduler.step()

    return history


def test_VGGNet(VGGNet, accelerator, train_loader, val_loader, num_classes) -> dict:
    print("=" * 60)
    print("Testing your CNN implementations on UVA Landmarks Dataset")
    print("=" * 60)

    # Test each architecture with fewer epochs for quick validation
    test_epochs = 5  # Increase to 20-30 for better results

    # Dictionary to store results
    results = {}

    # Part 1: Test VGGNet
    print("\n" + "="*60)
    print("Part 1: Testing VGGNet")
    print("="*60)
    try:
        vgg_model = VGGNet(num_classes=num_classes)
        print(f"VGGNet Parameters: {sum(p.numel() for p in vgg_model.parameters()):,}")
        vgg_history = train_model(accelerator, vgg_model, train_loader, val_loader, num_epochs=test_epochs)
        # {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        results['VGGNet'] = vgg_history['val_acc'][-1]
        plot_training_history(vgg_history, "VGGNet")
        return vgg_history
    except Exception as e:
        print(f"Error in VGGNet: {e}")
        results['VGGNet'] = 0
        return {}


def test_NiN(NiN, accelerator, train_loader, val_loader, num_classes) -> dict:
    # Part 2: Test NiN
    print("\n" + "="*60)
    print("Part 2: Testing Network in Network")
    print("="*60)
    results = {}
    try:
        nin_model = NiN(num_classes=num_classes)
        print(f"NiN Parameters: {sum(p.numel() for p in nin_model.parameters()):,}")
        test_epochs = 5
        nin_history = train_model(accelerator, nin_model, train_loader, val_loader, num_epochs=test_epochs)
        results['NiN'] = nin_history['val_acc'][-1]
        plot_training_history(nin_history, "NiN")
        return nin_history
    except Exception as e:
        print(f"Error in NiN: {e}")
        results['NiN'] = 0
        return {}


def test_GoogLeNet(GoogLeNet, accelerator, train_loader, val_loader, num_classes) -> dict:
    # Part 3: Test Inception
    print("\n" + "="*60)
    print("Part 3: Testing Inception Module")
    print("="*60)
    results = {}
    try:
        inception_model = GoogLeNet(num_classes=num_classes)
        print(f"GoogLeNet Parameters: {sum(p.numel() for p in inception_model.parameters()):,}")
        test_epochs = 5
        inception_history = train_model(accelerator, inception_model, train_loader, val_loader, num_epochs=test_epochs)
        results['Inception'] = inception_history['val_acc'][-1]
        plot_training_history(inception_history, "Inception")
        return inception_history
    except Exception as e:
        print(f"Error in Inception: {e}")
        results['Inception'] = 0
        return {}

def test_ResNet(ResNet, accelerator, train_loader, val_loader, num_classes) -> dict:
    # Part 4: Test ResNet
    print("\n" + "="*60)
    print("Part 4: Testing ResNet")
    print("="*60)
    results = {}
    try:
        resnet_model = ResNet(num_classes=num_classes)
        print(f"ResNet18 Parameters: {sum(p.numel() for p in resnet_model.parameters()):,}")
        test_epochs = 5
        resnet_history = train_model(accelerator, resnet_model, train_loader, val_loader, num_epochs=test_epochs)
        results['ResNet18'] = resnet_history['val_acc'][-1]
        plot_training_history(resnet_history, "ResNet18")
        return resnet_history
    except Exception as e:
        print(f"Error in ResNet: {e}")
        results['ResNet18'] = 0
        return {}


def test_transfer_learning(transfer_learning, accelerator, train_loader, val_loader, num_classes):
    # Part 5: Test Transfer Learning
    print("\n" + "="*60)
    print("Part 5: Testing Transfer Learning")
    print("="*60)

    # Test feature extraction
    results = {}
    test_epochs = 5
    try:
        print("\nTesting ResNet Feature Extraction (frozen backbone)...")
        pretrained_frozen = transfer_learning.get_pretrained_model('resnet18', num_classes=num_classes, feature_extract=True)
        frozen_history = train_model(
            accelerator, pretrained_frozen, train_loader, val_loader, num_epochs=test_epochs, feature_extract=True
        )
        results['ResNet_Transfer_Frozen'] = frozen_history['val_acc'][-1]
        plot_training_history(frozen_history, "ResNet Transfer Learning (Frozen)")
    except Exception as e:
        print(f"Error in ResNet Transfer Learning (Frozen): {e}")
        results['ResNet_Transfer_Frozen'] = 0

    try:
        print("\nTesting VGGNet Feature Extraction (frozen backbone)...")
        pretrained_frozen = transfer_learning.get_pretrained_model('vgg16', num_classes=num_classes, feature_extract=True)
        frozen_history = train_model(
            accelerator, pretrained_frozen, train_loader, val_loader, num_epochs=test_epochs, feature_extract=True
        )
        results['VGGNet_Transfer_Frozen'] = frozen_history['val_acc'][-1]
        plot_training_history(frozen_history, "VGGNet Transfer Learning (Frozen)")
    except Exception as e:
        print(f"Error in VGGNet Transfer Learning (Frozen): {e}")
        results['VGGNet_Transfer_Frozen'] = 0

    try:
        print("\nTesting MobileNet Feature Extraction (frozen backbone)...")
        pretrained_frozen = transfer_learning.get_pretrained_model(
            'mobilenet_v2', num_classes=num_classes, feature_extract=True
        )
        frozen_history = train_model(
            accelerator, pretrained_frozen, train_loader, val_loader, num_epochs=test_epochs, feature_extract=True
        )
        results['MobileNet_Transfer_Frozen'] = frozen_history['val_acc'][-1]
        plot_training_history(frozen_history, "MobileNet Transfer Learning (Frozen)")
    except Exception as e:
        print(f"Error in MobileNet Transfer Learning (Frozen): {e}")
        results['MobileNet_Transfer_Frozen'] = 0

    # Test fine-tuning
    try:
        print("\nTesting ResNet Fine-tuning (trainable backbone)...")
        pretrained_finetune = transfer_learning.get_pretrained_model('resnet18', num_classes=num_classes, feature_extract=False)
        finetune_history = train_model(
            accelerator, pretrained_finetune, train_loader, val_loader, num_epochs=test_epochs, lr=0.0001
        )
        results['ResNet_Transfer_Finetune'] = finetune_history['val_acc'][-1]
        plot_training_history(finetune_history, "ResNet Transfer Learning (Fine-tune)")
    except Exception as e:
        print(f"Error in ResNet Transfer Learning (Fine-tune): {e}")
        results['ResNet_Transfer_Finetune'] = 0

    try:
        print("\nTesting VGGNet Fine-tuning (trainable backbone)...")
        pretrained_finetune = transfer_learning.get_pretrained_model('vgg16', num_classes=num_classes, feature_extract=False)
        finetune_history = train_model(accelerator, pretrained_finetune, train_loader, val_loader,
                                      num_epochs=test_epochs, lr=0.0001)
        results['VGG_Transfer_Finetune'] = finetune_history['val_acc'][-1]
        plot_training_history(finetune_history, "VGGNet Transfer Learning (Fine-tune)")
    except Exception as e:
        print(f"Error in VGGNet Transfer Learning (Fine-tune): {e}")
        results['VGGNet_Transfer_Finetune'] = 0

    try:
        print("\nTesting MobileNet Fine-tuning (trainable backbone)...")
        pretrained_finetune = transfer_learning.get_pretrained_model(
            'mobilenet_v2', num_classes=num_classes, feature_extract=False
        )
        finetune_history = train_model(
            accelerator, pretrained_finetune, train_loader, val_loader, num_epochs=test_epochs, lr=0.0001
        )
        results['MobileNet_Transfer_Finetune'] = finetune_history['val_acc'][-1]
        plot_training_history(finetune_history, "MobileNet Transfer Learning (Fine-tune)")
    except Exception as e:
        print(f"Error in MobileNet Transfer Learning (Fine-tune): {e}")
        results['MobileNet_Transfer_Finetune'] = 0

    # Print summary of results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for model_name, accuracy in results.items():
        print(f"{model_name:20s}: {accuracy:.2f}%")

    best_model = max(results, key = lambda k: results[k])
    best_accuracy = results[best_model]
    print(f"\nBest Model: {best_model} with {best_accuracy:.2f}% validation accuracy")

    if best_accuracy > 94:
        print("\n🎉 Accuracy > 94%")
    else:
        print(f"\nCurrent best: {best_accuracy:.2f}% (Target: 94%)")

    print(results)
    return results


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model):
    """Calculate model size in MB (assuming float32 weights)."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_all_mb


def measure_inference_time(model, input_shape=(1, 3, 224, 224), num_runs=100):
    """Measure average inference time in milliseconds."""
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)

    # Warm up (important for accurate timing)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Synchronize if using CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time the inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs * 1000.0  # Convert to ms
    return avg_time


def estimate_flops(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    device: torch.device | None = None,
) -> int:
    """
    Estimate FLOPs (multiply-adds counted as 2 FLOPs) for a single forward pass.

    Counts:
      - nn.Conv2d: 2 * N * Hout * Wout * Cout * (Cin/groups) * Kh * Kw
      - nn.Linear: 2 * N * in_features * out_features

    Ignores (0 FLOPs):
      - activations, pooling, batchnorm, adds, etc. (usually small vs convs)
    """
    model_was_training = model.training
    model.eval()

    # Pick device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    flops_total = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def conv_hook(m: nn.Conv2d, inp, out):
        nonlocal flops_total
        # inp[0]: (N, Cin, Hin, Win)
        x = inp[0]
        N = x.shape[0]
        # out: (N, Cout, Hout, Wout)
        Hout, Wout = out.shape[-2], out.shape[-1]

        Cin = m.in_channels
        Cout = m.out_channels
        Kh, Kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        groups = m.groups

        # Multiply-adds per output element:
        # (Cin/groups) * Kh * Kw multiplies + same number adds => *2
        flops_per_out_elem = 2 * (Cin // groups) * Kh * Kw

        out_elems = N * Cout * Hout * Wout
        flops = out_elems * flops_per_out_elem

        # Optional: count bias adds (1 add per output element)
        if m.bias is not None:
            flops += out_elems

        flops_total += int(flops)

    def linear_hook(m: nn.Linear, inp, out):
        nonlocal flops_total
        x = inp[0]
        # x can be (N, in_features) or (..., in_features); flatten leading dims into batch
        in_features = m.in_features
        out_features = m.out_features
        batch = int(x.numel() // in_features)

        flops = 2 * batch * in_features * out_features
        if m.bias is not None:
            flops += batch * out_features  # bias adds
        flops_total += int(flops)

    # Register hooks
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    # Run a dummy forward
    dummy = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        _ = model.to(device)(dummy)

    # Cleanup
    for h in handles:
        h.remove()

    # Restore mode
    model.train(model_was_training)

    return flops_total


def plot_model_comparison(models_dict, device, input_shape = (1, 3, 224, 224)):
    """Compare efficiency metrics of different models."""
    _, axes = plt.subplots(2, 3, figsize=(18, 10))

    model_names = list(models_dict.keys())
    params_list = []
    size_list = []
    time_list = []
    flops_list = []

    for name, model in models_dict.items():
        total_params, _ = count_parameters(model)
        params_list.append(total_params / 1e6)  # Convert to millions
        size_list.append(get_model_size_mb(model))

        model = model.to(device)
        flops = estimate_flops(model, input_shape = input_shape, device = device)
        flops_list.append(flops / 1e9)

        time_list.append(measure_inference_time(model, input_shape = input_shape))

    # Plot 1: Parameters
    axes[0, 0].bar(model_names, params_list, color='blue', alpha=0.7)
    axes[0, 0].set_ylabel('Parameters (Millions)')
    axes[0, 0].set_title('Model Parameters Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, axis='y')

    # Plot 2: Model Size
    axes[0, 1].bar(model_names, size_list, color='green', alpha=0.7)
    axes[0, 1].set_ylabel('Size (MB)')
    axes[0, 1].set_title('Model Size on Disk')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, axis='y')

    # Plot 3: Inference Time
    axes[1, 0].bar(model_names, time_list, color='red', alpha=0.7)
    axes[1, 0].set_ylabel('Time (ms)')
    axes[1, 0].set_title('Inference Time (Lower is Better)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, axis='y')

    # Plot 4: FLOPs (replaces your scatter)
    axes[1, 1].bar(model_names, flops_list, alpha = 0.7)
    axes[1, 1].set_ylabel("FLOPs (GFLOPs)")
    axes[1, 1].set_title("Compute Cost per Forward (Lower is Better)")
    axes[1, 1].tick_params(axis = 'x', rotation = 45)
    axes[1, 1].grid(True, axis = 'y')

    # Plot 5: Efficiency Score
    axes[1, 2].scatter(params_list, time_list, s=100, alpha=0.7)
    for i, name in enumerate(model_names):
        axes[1, 2].annotate(name, (params_list[i], time_list[i]), fontsize=8, ha='right')
    axes[1, 2].set_xlabel('Parameters (Millions)')
    axes[1, 2].set_ylabel('Inference Time (ms)')
    axes[1, 2].set_title('Efficiency Trade-off (Lower-left is Better)')
    axes[1, 2].grid(True)

    axes[0, 2].axis("off")

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()


def test_assignment_extension(MobileNet, accelerator, train_loader, val_loader, num_classes):   
    print("="*80)
    print("Testing Your Efficient Architecture Implementation")
    print("="*80)

    # Test your implementations
    try:
        # Test DepthwiseSeparableConv
        print("\n1. Testing DepthwiseSeparableConv...")
        dw_conv = MobileNet.DepthwiseSeparableConv(32, 64).to(accelerator)
        test_input = torch.randn(1, 32, 56, 56, device = accelerator)
        output_of_DepthwiseSeparableConv = dw_conv(test_input)
        numpy_array = output_of_DepthwiseSeparableConv.cpu().detach().numpy()
        # np.save("output_of_DepthwiseSeparableConv.npy", numpy_array)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output_of_DepthwiseSeparableConv.shape}")
        print(f"   ✓ DepthwiseSeparableConv working!")
    except Exception as e:
        print(f"   ✗ Error in DepthwiseSeparableConv: {e}")
        numpy_array = None

    try:
        # Test InvertedResidual
        print("\n2. Testing InvertedResidual...")
        inv_res = MobileNet.InvertedResidual(32, 32, stride=1, expand_ratio=6).to(accelerator)
        test_input = torch.randn(1, 32, 56, 56, device = accelerator)
        output = inv_res(test_input)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   ✓ InvertedResidual working!")
    except Exception as e:
        print(f"   ✗ Error in InvertedResidual: {e}")

    try:
        # Test MobileNet
        print("\n3. Testing MobileNet...")
        mobilenet = MobileNet.MobileNet(num_classes=num_classes).to(accelerator)
        test_input = torch.randn(1, 3, 224, 224, device = accelerator)
        output = mobilenet(test_input)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")

        # Analyze model
        total_params, trainable_params = count_parameters(mobilenet)
        model_size = get_model_size_mb(mobilenet)
        inf_ms = measure_inference_time(mobilenet.to(accelerator), input_shape = (1, 3, 224, 224))
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: {model_size:.2f} MB")
        print(f"    inference (ms): {inf_ms:.2f}")
        print(f"   ✓ MobileNet working!")

    except Exception as e:
        print(f"   ✗ Error in MobileNet: {e}")

    # Compare with other models
    print("\n" + "="*80)
    print("Model Comparison")
    print("="*80)

    try:
        # Create models for comparison
        models_to_compare = {
            'Your MobileNet': MobileNet.MobileNet(num_classes=num_classes),
            'ResNet18': models.resnet18(num_classes=num_classes),
            'Torchvision MobileNet initialized randomly': models.mobilenet_v2(weights=None, num_classes=num_classes)
        }

        # Compare models
        for name, model in models_to_compare.items():
            model = model.to(accelerator)
            total_params, _ = count_parameters(model)
            size_mb = get_model_size_mb(model)
            flops = estimate_flops(model, input_shape = (1, 3, 224, 224), device = accelerator)
            print(f"{name:20s}: {total_params/1e6:.2f}M params, {size_mb:.2f} MB, {flops/1e9:.2f} GFLOPs")

        # Visualize comparison
        plot_model_comparison(models_to_compare, accelerator)

    except Exception as e:
        print(f"Error in model comparison: {e}")

    # Train your model (optional - takes time)
    print("\n" + "="*80)
    print("Training Your MobileNet")
    print("="*80)

    try:
        model = MobileNet.MobileNet(num_classes=num_classes, dropout_prob=0.1).to(accelerator)

        history = train_model(accelerator, model, train_loader, val_loader, num_epochs = 5, lr = 0.001, feature_extract = False, step_size = 3, gamma = 0.01)
        plot_training_history(history, "MobileNet_per_extension")

        print(f"\nFinal Validation Accuracy: {history['val_acc'][-1]:.2f}%")
        if history['val_acc'][-1] > 80:
            print("✓ Great job! Your model achieves good accuracy while being efficient!")
        elif history['val_acc'][-1] > 70:
            print("✓ Good start! Try fine-tuning hyperparameters or training longer.")
        else:
            print("Keep working! Check your implementation and try different settings.")

    except Exception as e:
        print(f"Error during training: {e}")
        history = {}

    return numpy_array, history


def main() -> None:
    batch_size = 32
    data_dir = "dataset/"
    num_classes = 18
    num_workers = 0
    seed = 42
    accelerator, base_ds, train_idx, val_idx = set_up(seed = seed, data_dir = data_dir)
    train_loader_150, val_loader_150 = make_loaders(
        base_ds, train_idx, val_idx, seed = seed, batch_size = batch_size,
        num_workers = num_workers, image_size = (150, 150), train_aug = "resize_flip"
    )
    # VGGNet = import_module("VGGNet")
    # NiN = import_module("NiN")
    # GoogLeNet = import_module("GoogLeNet")
    # ResNet = import_module("ResNet")
    # transfer_learning = import_module("transfer_learning")
    # test_VGGNet(VGGNet.VGGNet, accelerator, train_loader_150, val_loader_150, num_classes)
    # test_NiN(NiN.NiN, accelerator, train_loader_150, val_loader_150, num_classes)
    # test_GoogLeNet(GoogLeNet.GoogLeNet, accelerator, train_loader_150, val_loader_150, num_classes)
    # test_ResNet(ResNet.ResNet, accelerator, train_loader_150, val_loader_150, num_classes)
    # test_transfer_learning(transfer_learning, accelerator, train_loader_150, val_loader_150, num_classes)
    train_loader_224, val_loader_224 = make_loaders(base_ds, train_idx, val_idx, seed = seed, batch_size = batch_size, num_workers = num_workers, image_size = 224, train_aug = "random_resized_crop")
    MobileNet = import_module("MobileNet")
    test_assignment_extension(MobileNet, accelerator, train_loader_224, val_loader_224, num_classes)


if __name__ == '__main__':
    main()