import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from memory_dataset import MemoryData
from KeyframeClassifier import KeyframeClassifier_big_lowdim, KeyframeClassifier_resnet, KeyframeClassifier_small_lowdim, KeyframeClassifier_no_lowdim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets_config = [
    {"name": "0422_driver_box_heightline_smalllowdim_cp10", "zarr_path": "../dataset/0422_driver_heightline.zarr", "model_type": "vit_small_lowdim", "oversample_factor": 10}, 
    {"name": "0422_driver_box_lookingdown_smalllowdim_cp10", "zarr_path": "../dataset/0422_driver_lookingdown.zarr", "model_type": "vit_small_lowdim", "oversample_factor": 10}, 
]

model_map = {
    "resnet": KeyframeClassifier_resnet,
    "vit_no_lowdim": KeyframeClassifier_no_lowdim,
    "vit_big_lowdim": KeyframeClassifier_big_lowdim,
    "vit_small_lowdim": KeyframeClassifier_small_lowdim,
}

num_epochs = 300
learning_rate = 1e-5
weight_decay = 1e-5
batch_size = 1024
num_workers = 12
save_every_epochs = 50
class_weights_list = [10, 1]
class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss()
weight_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

for dataset_info in datasets_config:
    dataset_name = dataset_info["name"]
    zarr_path = dataset_info["zarr_path"]
    model_type_str = dataset_info["model_type"].lower()  
    oversample = dataset_info.get("oversample_factor") 

    print(f"Starting training for dataset: {dataset_name} with model: {model_type_str}")

    train_data = MemoryData(zarr_path=zarr_path, train=True, oversample_factor=oversample)
    test_data = MemoryData(zarr_path=zarr_path, train=False)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if model_type_str in model_map:
        model = model_map[model_type_str]()
    else:
        print(f"Warning: Model type '{model_type_str}' not found. Using default model.")
        model = model_map["default"]()

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    log_dir = f"./logs_train/{dataset_name}_{model_type_str}/"
    writer = SummaryWriter(log_dir)

    save_dir = f"./saved_models/{dataset_name}_{model_type_str}/"
    os.makedirs(save_dir, exist_ok=True)

    total_train_step = 0
    total_test_step = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0

        for img, pose_velocity, label in train_data_loader:
            img, pose_velocity, label = img.to(device), pose_velocity.to(device), label.to(device).long()

            optimizer.zero_grad()
            logits = model(img, pose_velocity)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == label).sum().item()
            total_correct += correct

            total_train_step += 1
            if total_train_step % 100 == 0:
                # print(f"[{dataset_name} - {model_type_str}] Epoch {epoch+1}, Train Step {total_train_step}, Loss: {loss.item():.4f}")
                writer.add_scalar(f"/Train/Batch_loss", loss.item(), total_train_step)

        train_accuracy = total_correct / len(train_data)
        avg_train_loss = total_loss / len(train_data_loader)
        print(f"[{dataset_name} - {model_type_str}] Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        writer.add_scalar(f"/Train/train_accuracy", train_accuracy, epoch)
        writer.add_scalar(f"/Train/train_loss_epoch_avg", avg_train_loss, epoch)

        model.eval()
        total_test_loss = 0
        total_correct = 0
        with torch.no_grad():
            for img, pose_velocity, label in test_data_loader:
                img, pose_velocity, label = img.to(device), pose_velocity.to(device), label.to(device).long()
                logits = model(img, pose_velocity)
                loss = criterion(logits, label)
                total_test_loss += loss.item() * img.size(0)
                predictions = torch.argmax(logits, dim=1)
                correct = (predictions == label).sum().item()
                total_correct += correct

        test_loss_avg = total_test_loss / len(test_data)
        test_accuracy = total_correct / len(test_data)
        print(f"[{dataset_name} - {model_type_str}] Epoch {epoch+1}, Test Loss: {test_loss_avg:.4f}, Accuracy: {test_accuracy:.4f}")
        writer.add_scalar(f"/Test/test_accuracy", test_accuracy, epoch)
        writer.add_scalar(f"/Test/test_loss_epoch_avg", test_loss_avg, epoch)

        if (epoch + 1) % save_every_epochs == 0:
            checkpoint_path = os.path.join(save_dir, f"{dataset_name}_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": total_loss,
                "train_accuracy": train_accuracy,
                "test_loss": total_test_loss,
                "test_accuracy": test_accuracy
            }, checkpoint_path)
            print(f"[{dataset_name} - {model_type_str}] Model saved at {checkpoint_path}")

    writer.close()
    print(f"Finished training for dataset: {dataset_name} with model: {model_type_str}\n")

print("Training for all datasets completed.")