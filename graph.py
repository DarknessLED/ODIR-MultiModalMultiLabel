import torch
from matplotlib import pyplot as plt

model_name = "ModalClassifier"

# 检查点路径
checkpoint_path = f"{model_name}.pth"
checkpoint = torch.load(checkpoint_path)
train_loss_history = checkpoint['train_loss_history']
train_total_loss_history = checkpoint['train_total_loss_history']
valid_loss_history = checkpoint['valid_loss_history']
train_acc_history = checkpoint['train_acc_history']
valid_acc_history = checkpoint['valid_acc_history']
train_pre_history = checkpoint['train_pre_history']
valid_pre_history = checkpoint['valid_pre_history']
train_rec_history = checkpoint['train_rec_history']
valid_rec_history = checkpoint['valid_rec_history']
train_f1_history = checkpoint['train_f1_history']
valid_f1_history = checkpoint['valid_f1_history']
epoch = checkpoint['epoch']

train_loss_history_cpu = [loss.detach().cpu().numpy() for loss in train_loss_history]
train_total_loss_history_cpu = [loss.detach().cpu().numpy() for loss in train_total_loss_history]
valid_loss_history_cpu = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in valid_loss_history]
train_acc_history_cpu = [acc.detach().cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in train_acc_history]
valid_acc_history_cpu = [acc for acc in valid_acc_history]
train_pre_history_cpu = [pre if isinstance(pre, torch.Tensor) else pre for pre in train_pre_history]
valid_pre_history_cpu = [pre if isinstance(pre, torch.Tensor) else pre for pre in valid_pre_history]
train_rec_history_cpu = [rec if isinstance(rec, torch.Tensor) else rec for rec in train_rec_history]
valid_rec_history_cpu = [rec if isinstance(rec, torch.Tensor) else rec for rec in valid_rec_history]
train_f1_history_cpu = [f1 if isinstance(f1, torch.Tensor) else f1 for f1 in train_f1_history]
valid_f1_history_cpu = [f1 if isinstance(f1, torch.Tensor) else f1 for f1 in valid_f1_history]

# print(len(train_loss_history))
# print(epoch)

# Loss 曲线
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 3, 1)
# plt.plot(train_loss_history_cpu, label='Train Step Loss', color='blue')
# plt.xlabel('Step')
# plt.ylabel('Loss')
# plt.legend()

# Loss 曲线
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.plot(train_total_loss_history_cpu, label='Train Loss', color='blue')
# plt.plot(valid_loss_history_cpu, label='Valid Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 3, 2)
# plt.plot(train_total_loss_history_cpu, label='Train Loss', color='blue')
plt.plot(valid_loss_history_cpu, label='Valid Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy 曲线
plt.subplot(2, 3, 3)
plt.plot(train_acc_history, label='Train Accuracy', color='blue')
plt.plot(valid_acc_history, label='Valid Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Precision 曲线
plt.subplot(2, 3, 4)
plt.plot(train_pre_history_cpu, label='Train Precision', color='blue')
plt.plot(valid_pre_history_cpu, label='Valid Precision', color='red')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

# Recall 曲线
plt.subplot(2, 3, 5)
plt.plot(train_rec_history_cpu, label='Train Recall', color='blue')
plt.plot(valid_rec_history_cpu, label='Valid Recall', color='red')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

# F1 Score 曲线
plt.subplot(2, 3, 6)
plt.plot(train_f1_history_cpu, label='Train F1 Score', color='blue')
plt.plot(valid_f1_history_cpu, label='Valid F1 Score', color='red')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()
