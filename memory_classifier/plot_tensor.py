from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os

log_dirs = {
    'Resnet+Gripper+DA': 'logs_train/0417_driver_box_resnet_cp10_resnet/',
    'ViT+Gripper': 'logs_train/0417_driver_box_vit_small_lowdim_nocp_vit_small_lowdim/',
    'ViT+DA': 'logs_train/0417_driver_box_vit_no_lowdim_cp10_vit_no_lowdim/',
    'ViT+Gripper+DA': 'logs_train/0417_driver_box_vit_small_lowdim_cp10_vit_small_lowdim/',
}

tag_name = '/Train/train_loss_epoch_avg'
output_path = 'merged_train_loss.png'

plt.figure(figsize=(10, 6))

for label, log_dir in log_dirs.items():
    try:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        if tag_name not in ea.Tags()['scalars']:
            print(f"[WARNING] Tag '{tag_name}' not found in {log_dir}")
            continue

        events = ea.Scalars(tag_name)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        plt.plot(steps, values, label=label)
    except Exception as e:
        print(f"[ERROR] Failed to process {log_dir}: {e}")

plt.xlabel("Epoch")
plt.ylabel("Train Loss")
# plt.ylim([0.85,1.05])
plt.title(f"Average Train Loss Per Epoch Across Models")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"[✓] Saved merged plot to: {output_path}")
