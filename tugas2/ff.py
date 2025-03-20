import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd 



train_dataset_path = "./Dataset/TRAINING"
test_dataset_path = "./Dataset/TESTING"

class ImageTensorDataset(Dataset):
    def __init__(self, dataset_path, size=(224, 224)):
        self.image_tensors = []
        self.labels = []
        self.class_to_idx = {}
        current_label = 0

        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

        # Pastikan dataset_path ada
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path '{dataset_path}' not found.")
            self.image_tensors = torch.empty(0, 3, *size)  # Tensor kosong untuk citra
            self.labels = torch.empty(0, dtype=torch.long)  # Tensor kosong untuk label
            return

        for person in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person)
            if not os.path.isdir(person_path):
                continue

            if person not in self.class_to_idx:
                self.class_to_idx[person] = current_label
                current_label += 1

            for img_file in os.listdir(person_path):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(person_path, img_file)
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img_tensor = transform(img)
                        self.image_tensors.append(img_tensor)
                        self.labels.append(self.class_to_idx[person])
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")

        # Menghindari error jika tidak ada gambar yang dimuat
        if len(self.image_tensors) > 0:
            self.image_tensors = torch.stack(self.image_tensors)
            self.labels = torch.tensor(self.labels, dtype=torch.long)
        else:
            self.image_tensors = torch.empty(0, 3, *size)  # Tensor kosong dengan shape (0, 3, 224, 224)
            self.labels = torch.empty(0, dtype=torch.long)  # Tensor kosong untuk label

    def __len__(self):
        return len(self.image_tensors)
    
    def __getitem__(self, idx):
        if idx >= len(self.image_tensors):  # Hindari IndexError
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.image_tensors)}")
        return self.image_tensors[idx], self.labels[idx]

class ANNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ANNTrainer:
    def __init__(self, model, train_loader, test_loader, lr=0.01, optimizer_type="SGD", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        if optimizer_type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer_type == "MiniBatchGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_type == "GD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0)
        else:
            raise ValueError("Optimizer type not recognized")

    def train(self, epochs=10):
        self.model.train()
        training_errors = []
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.view(images.size(0), -1)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            training_errors.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        return training_errors
    
    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        all_pred = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.view(images.size(0), -1)

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_pred.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        print(f"Testing Accuracy: {accuracy:.2f}%")
        return accuracy,all_pred,all_labels


# Load Dataset
train_dataset = ImageTensorDataset(train_dataset_path)
test_dataset = ImageTensorDataset(test_dataset_path)

print(f"Jumlah data pelatihan: {len(train_dataset)}")
print(f"Jumlah data pengujian: {len(test_dataset)}")

# Buat DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Jumlah batch pelatihan: {len(train_loader)}")
print(f"Jumlah batch pengujian: {len(test_loader)}")

# Inisialisasi Model
input_size = 224 * 224 * 3  # Flatten dari citra RGB ukuran 224x224
num_classes = len(train_dataset.class_to_idx)  # Jumlah kelas
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}  # Mapping indeks ke nama kelas

print(f"Input size: {input_size}")
print(f"Number of classes: {num_classes}")
print(f"Index to class mapping: {idx_to_class}")

# set up params ANN

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer_type = "SGD"

model = ANNModel(input_size, num_classes).to(device)
trainer=ANNTrainer (model, train_loader, test_loader, optimizer_type=optimizer_type)

epochs=100

model.train()
training_errors = trainer.train(epochs=epochs)
print(f"Using device: {device}")
print(f"Using optimizer: {optimizer_type}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy,all_preds,all_labels = trainer.test()

test_results = []
model.eval()

with torch.no_grad():
    for i in range(len(train_dataset)):
        img_tensor, label = train_dataset[i]
        img_tensor = img_tensor.to(device)
        img_tensor = img_tensor.reshape(-1, input_size)
        output = model(img_tensor)
        _,predicted = torch.max(output, 1)
        pred_label = idx_to_class[predicted.item()]
        true_label = idx_to_class[label.item()]
        file_name = f"train_image_{i}.png"
        test_results.append([file_name, pred_label, true_label])
# Simpan hasil testing ke CSV
pd.DataFrame(test_results, columns=["File Name", "Predicted Label", "Actual Label"]).to_csv("test_results.csv", index=False)

# Evaluasi Model dengan Data Testing
accuracy = accuracy_score(all_labels,all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

print(f"Evaluation Metrics:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")

# Plot Error Training
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(training_errors) + 1), training_errors, marker='o', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Error over Epochs")
plt.show()

print("Hasil prediksi data training disimpan di train_results.csv")
print("Hasil prediksi data testing disimpan di test_results.csv")

# Menampilkan Gambar Hasil Testing
num_images = 20
fnum_images = min(num_images, len(test_dataset))  # Pastikan tidak melebihi jumlah dataset
rows = (num_images // 5) + (num_images % 5 > 0)  # Mengatur jumlah baris agar tidak terlalu panjang
cols = min(num_images, 5)  # Maksimal 5 gambar per baris

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

if num_images == 1:
    axes = [axes]  # Jika hanya satu gambar, jadikan list agar tetap bisa diakses dengan indeks

axes = axes.flatten()  # Meratakan array agar mudah diakses dalam loop

for i in range(num_images):
    img_tensor, label = test_dataset[i]
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # Konversi tensor ke numpy untuk ditampilkan
    
    img_tensor = img_tensor.to(device).reshape(-1, input_size)
    output = model(img_tensor)
    predicted = torch.argmax(output, dim=1)

    pred_label = idx_to_class[predicted.item()]
    true_label = idx_to_class[label.item()]

    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(f"Actual: {true_label}\nPred: {pred_label}", fontsize=8)

# Sembunyikan axis untuk subplot yang tidak terpakai
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

