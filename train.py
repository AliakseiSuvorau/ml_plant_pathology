import timm
import torch
from tqdm import tqdm
import utils
import config
from os.path import join
import numpy as np

# Preparing data
train_loader, val_loader = utils.get_data_loader(
    data_path=config.data_path,
    data_file=config.train_data_filename,
    phase="train",
    train_val_split=0.2,
    batch_size=32
)

# Preparing model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = timm.create_model(config.model_name, pretrained=True, num_classes=4).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_score = 0.0

# Transfer learning
for epoch in range(config.num_epochs):
    # Training
    model.train()
    train_loss = 0
    metrics = {
        'train_roc_auc': [],
        'val_roc_auc': []
    }

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        outputs = torch.softmax(outputs, dim=-1)
        roc_auc = utils.mean_colwise_roc_auc(labels, outputs)
        metrics['train_roc_auc'].append(roc_auc.cpu().numpy())

        # Backward pass
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        val_preds = []
        val_targets = []

        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)

            # Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            roc_auc = utils.mean_colwise_roc_auc(labels, outputs)
            metrics['val_roc_auc'].append(roc_auc.cpu().numpy())

            val_loss += loss.item()

            val_preds.append(outputs)
            val_targets.append(labels)

    train_roc_auc_score = np.mean(metrics['train_roc_auc'])
    val_roc_auc_score = np.mean(metrics['val_roc_auc'])
    print(f"\nEpoch {epoch + 1} | Train roc auc: {train_roc_auc_score:.4f} | Val roc auc: {val_roc_auc_score:.4f}")

    if val_roc_auc_score > best_score:
        best_score = val_roc_auc_score
        torch.save(model.state_dict(), str(join(config.data_out_dir_path, config.checkpoint_filename)))
