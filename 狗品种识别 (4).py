from email.headerregistry import DateHeader
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import shutil
# 图片预处理
class DogBreedDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 更新品种映射
        self.breed_to_idx = {
            'labrador': 0,
            'german_shepherd': 1,
            'golden_retriever': 2,
            'french_bulldog': 3,
            'poodle': 4,
            'shiba_inu': 5,
            'husky': 6,
            'border_collie': 7,
            'bichon_frise': 8,
            'corgi': 9,
            'samoyed': 10,
            'doberman': 11,
            'great_dane': 12,
            'st_bernard': 13,
            'yorkshire': 14
        }
        
        # 中英文品种名映射
        self.breed_names = {
            'labrador': '拉布拉多',
            'german_shepherd': '德国牧羊犬',
            'golden_retriever': '金毛寻回犬',
            'french_bulldog': '法国斗牛犬',
            'poodle': '贵宾犬',
            'shiba_inu': '柴犬',
            'husky': '哈士奇',
            'border_collie': '边境牧羊犬',
            'bichon_frise': '比熊犬',
            'corgi': '柯基',
            'samoyed': '萨摩耶',
            'doberman': '杜宾犬',
            'great_dane': '大丹犬',
            'st_bernard': '圣伯纳犬',
            'yorkshire': '约克夏'
        }
        
        self.images = []
        self.labels = []
        
        train_dir = os.path.join(data_dir, 'train')
        if os.path.exists(train_dir):
            self.classes = sorted([d for d in os.listdir(train_dir) 
                                 if os.path.isdir(os.path.join(train_dir, d))])
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(data_dir, split)
            if not os.path.exists(split_dir):
                continue
                
            for class_name in self.classes:
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                    
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        if os.path.isfile(img_path):
                            self.images.append(img_path)
                            self.labels.append(self.class_to_idx[class_name])
        
        print(f"找到 {len(self.images)} 张图片，{len(self.classes)} 个品种")
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"错误：无法加载图片 {img_path}: {str(e)}")
            image = torch.zeros((3, 224, 224))
            return image, label

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load_data(self):
        """加载数据集并进行基本验证"""
        dataset = DogBreedDataset(self.data_path)
        total_size = len(dataset)
        
        if total_size == 0:
            raise ValueError("数据集为空！")
            
        print(f"总共加载了 {total_size} 张图片")
        print(f"包含 {len(dataset.classes)} 个狗品种类别")
        return dataset
    
    def split_data(self, dataset, train_ratio=0.7, val_ratio=0.15):
        """划分数据集为训练集、验证集和测试集"""
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_data_loaders(self, train_dataset, val_dataset, test_dataset, batch_size=32):
        """创建数据加载器"""
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
    def get_data_loaders(self, batch_size=32):
        """完整的数据处理流程"""
        # 加载数据
        dataset = self.load_data()
        
        # 划分数据集
        train_dataset, val_dataset, test_dataset = self.split_data(dataset)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_dataset, val_dataset, test_dataset, batch_size
        )
        
        return train_loader, val_loader, test_loader, len(dataset.classes)

def setup_data_directory():
    """设置数据目录结构"""
    base_dir = "data"
    dog_breeds_dir = os.path.join(base_dir, "dog_breeds")
    
    # 创建主目录
    os.makedirs(dog_breeds_dir, exist_ok=True)
    
    # 检查现有数据
    if os.path.exists(base_dir):
        # 查找所有图片文件
        image_files = []
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        if image_files:
            print(f"找到 {len(image_files)} 张图片")
            
            # 如果图片都在一个文件夹中，需要按品种分类
            if not any(os.path.isdir(os.path.join(dog_breeds_dir, d)) for d in os.listdir(dog_breeds_dir)):
                print("警告：图片未按品种分类。请确保将图片放在对应品种的子文件夹中。")
                print("目录结构应该是：")
                print("data/dog_breeds/")
                print("    ├── 品种1/")
                print("    │   ├── image1.jpg")
                print("    │   └── image2.jpg")
                print("    └── 品种2/")
                print("        ├── image1.jpg")
                print("        └── image2.jpg")
        else:
            print("未找到图片文件！")
            print("请将狗品种图片数据放在 data/dog_breeds/ 目录下的对应品种子文件夹中")
    
    return dog_breeds_dir

if __name__ == "__main__":
    # 首先设置数据目录
    data_path = setup_data_directory()
    print(f"\n数据目录: {data_path}")
    
    try:
        # 创建数据处理器实例
        processor = DataProcessor(data_path)
        
        # 获取数据加载器
        train_loader, val_loader, test_loader, num_classes = processor.get_data_loaders(batch_size=32)
        
        # 测试数据加载器
        print("\n测试数据加载器:")
        print(f"训练集批次数量: {len(train_loader)}")
        print(f"验证集批次数量: {len(val_loader)}")
        print(f"测试集批次数量: {len(test_loader)}")
        print(f"类别数量: {num_classes}")
        
        # 获取一个批次的数据进行测试
        images, labels = next(iter(train_loader))
        print(f"\n数据批次形状:")
        print(f"图像张量形状: {images.shape}")
        print(f"标签张量形状: {labels.shape}")
        
    except Exception as e:
        print(f"错误: {str(e)}")