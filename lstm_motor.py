import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange


# 設定參數
input_size = 10  # 輸入特徵數量
hidden_size = 128  # LSTM 隱藏層大小
num_layers = 2  # LSTM 層數
num_classes = 6  # 類別數量
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# 定義資料集類別
class RideTrackDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 定義 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # print("x.shape: ", x.shape)  # x shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # x shape: (batch_size, input_size) -> (batch_size, 1, input_size)

        out, _ = self.lstm(x)  # out shape: (batch_size, 1, hidden_size)
        # print("out.shape: ", out.shape)
        out = out.squeeze(1)  # out shape: (batch_size, hidden_size)

        out = self.fc(out)  # out shape: (batch_size, num_classes)
        # print("out.shape: ", out.shape)
        return out


# 讀取 CSV 資料
data = pd.read_csv('mixed_data/mixed_train.csv')  # 替換成你的 CSV 檔案路徑

# 選擇特徵和標籤
features = ['Z-axis Angular Velocity', 'Yaw (deg)', 'Y-axis Acceleration', 'Z-axis Acceleration',
            'Pitch (deg)', 'X-axis Angular Velocity', 'Y-axis Angular Velocity', 'X-axis Acceleration',
            'Roll (deg)', 'Vehicle Speed']

target = 'Action'

# 移除空標籤的資料
data = data[data[target] > ''] 

X = data[features].values
y = data[target].values

print(f'X.shape: {X.shape}, y.shape: {y.shape}')  # X.shape: (47836, 10), y.shape: (47836,)

# 使用 LabelEncoder 將標籤編碼
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 分割訓練集為訓練集和驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# 建立資料集和 DataLoader
train_dataset = RideTrackDataset(X_train, y_train)
val_dataset = RideTrackDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 建立模型、損失函數和優化器
model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 訓練模型
train_losses = []  # 儲存訓練損失值
val_losses = []  # 儲存驗證損失值
val_accuracies = []  # 儲存驗證準確率

for epoch in range(num_epochs):
    epoch_train_loss = 0.0
    # 使用 tqdm 包裹 train_loader 並將其儲存到變數 pbar
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False) as pbar:
        for i, (inputs, labels) in enumerate(pbar):
            # 前向傳播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            # 使用 pbar.set_postfix 更新顯示訊息
            pbar.set_postfix({'Loss': '{:.4f}'.format(loss.item())})

    # 計算每個 epoch 的平均訓練損失值
    avg_epoch_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_epoch_train_loss)

    # 驗證模型
    model.eval()  # 將模型設定為評估模式
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    model.train()  # 將模型設定回訓練模式

# 儲存模型
torch.save(model.state_dict(), './lstm_self_attn_motor_models/lstm_ridetrack_model.pth')

# 繪製損失值折線圖
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# 載入模型
model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(torch.load('./lstm_self_attn_motor_models/lstm_ridetrack_model.pth'))  # 載入模型權重

# 讀取 CSV 資料
data = pd.read_csv('./20240708_data/20240708_Eric_sliding_window_merged.csv')

# 選擇特徵和標籤
features = ['Z-axis Angular Velocity', 'Yaw (deg)', 'Y-axis Acceleration', 'Z-axis Acceleration',
            'Pitch (deg)', 'X-axis Angular Velocity', 'Y-axis Angular Velocity', 'X-axis Acceleration',
            'Roll (deg)', 'Vehicle Speed']

target = 'Action'

# 移除空標籤的資料
data = data[data[target] > ''] 

X = data[features].values
y = data[target].values

# 使用 LabelEncoder 將標籤編碼
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 建立測試資料集和 DataLoader
test_dataset = RideTrackDataset(X, y)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 測試模型
model.eval()  # 將模型設定為評估模式
with torch.no_grad():
    all_preds = []
    all_labels = []
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# 計算混淆矩陣和正確率
cm = confusion_matrix(all_labels, all_preds)
accuracy = accuracy_score(all_labels, all_preds)

# 輸出混淆矩陣(文字模式)
# print("Confusion Matrix:")
# print(cm)

# 繪製混淆矩陣
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)  # 設定 x 和 y 軸的標籤
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.yticks(rotation=0)  # 調整 y 軸標籤為水平
plt.xticks(rotation=45)  # 調整 x 軸標籤傾斜 45 度
plt.title('Confusion Matrix')
plt.show()

# print(f"Accuracy: {accuracy:.4f}")  # 輸出平均正確率

# 計算每個類別的準確率
class_accuracies = []
for class_id in range(num_classes):
    true_positives = cm[class_id, class_id]
    total_samples = sum(cm[class_id, :])
    if total_samples > 0:
        class_accuracy = true_positives / total_samples * 100
    else:
        class_accuracy = 0  # 避免除以零
    class_accuracies.append(class_accuracy)

# 繪製正確率長條圖
colors = plt.cm.viridis(np.linspace(0, 1, num_classes))

plt.figure(figsize=(10, 6))
plt.bar(label_encoder.classes_, class_accuracies, color=colors)
plt.axhline(y=accuracy * 100, color='red', linestyle='--')
plt.text(num_classes-0.5, accuracy * 100 + 1, f'Accuracy (Total): {accuracy*100:.2f}', color='black', ha='right', fontsize=16)  # plt.text(x座標, y座標, ...)
plt.xlabel('Action')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy analysis of different behaviors')
plt.xticks(rotation=45)
plt.show()