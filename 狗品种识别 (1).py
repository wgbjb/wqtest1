import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# 训练集
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DogBreedDataset
from models.model import DogBreedClassifier
import matplotlib.pyplot as plt
import numpy as np

class TrainingVisualizer:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # 设置中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
        except:
            print("警告: SimHei字体不可用，尝试其他字体...")
            try:
                # 尝试其他中文字体
                import matplotlib.font_manager as fm
                chinese_fonts = ['Microsoft YaHei', 'WenQuanYi Micro Hei', 'SimSun', 'NSimSun']
                for font in chinese_fonts:
                    if any([f.name == font for f in fm.fontManager.ttflist]):
                        plt.rcParams['font.sans-serif'] = [font]
                        plt.rcParams['axes.unicode_minus'] = False
                        break
            except:
                print("警告: 无法设置中文字体，将使用英文显示")
        
    def update(self, train_loss, val_loss, val_acc, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        
    def plot_metrics(self, save_dir='plots'):
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建图形和子图
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        epochs = range(1, len(self.train_losses) + 1)
        
        # 绘制损失曲线
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss', fontsize=12)
        ax1.set_xlabel('Epoch', fontsize=10)
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.legend(fontsize=10)
        ax1.grid(True)
        
        # 绘制准确率曲线
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy', fontsize=12)
        ax2.set_xlabel('Epoch', fontsize=10)
        ax2.set_ylabel('Accuracy (%)', fontsize=10)
        ax2.legend(fontsize=10)
        ax2.grid(True)
        
        # 绘制学习率曲线
        ax3.plot(epochs, self.learning_rates, 'y-', label='Learning Rate')
        ax3.set_title('Learning Rate', fontsize=12)
        ax3.set_xlabel('Epoch', fontsize=10)
        ax3.set_ylabel('Learning Rate', fontsize=10)
        ax3.legend(fontsize=10)
        ax3.grid(True)
        ax3.set_yscale('log')  # 使用对数刻度
        
        # 调整子图间距
        plt.tight_layout()
        
        # 保存图像（使用更高DPI以提高清晰度）
        plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

def train_model(data_dir, num_epochs=50, batch_size=32, learning_rate=0.001):
    # 创建可视化器
    visualizer = TrainingVisualizer()
    
    # 创建数据集
    dataset = DogBreedDataset(data_dir)
    total_size = len(dataset)
    
    # 划分数据集: 70% 训练, 15% 验证, 15% 测试
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = DogBreedClassifier(num_classes=len(dataset.classes)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 计算平均损失和准确率
        avg_train_loss = running_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        val_acc = 100 * correct / total
        current_lr = scheduler.get_last_lr()[0]
        
        # 更新可视化器
        visualizer.update(avg_train_loss, avg_val_loss, val_acc, current_lr)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # 每5个epoch绘制一次图表
        if (epoch + 1) % 5 == 0:
            visualizer.plot_metrics()
    
    # 训练结束后绘制最终图表
    visualizer.plot_metrics()
    
    # 测试集评估
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    
    print("\n开始测试集评估...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    print(f'\n测试集结果:')
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    return model, test_acc, visualizer

if __name__ == "__main__":
    model, test_accuracy, visualizer = train_model("data/")