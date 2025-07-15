import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# 预测结果
import torch
from PIL import Image
import os
from tqdm import tqdm
from models.model import DogBreedClassifier
from dataset import DogBreedDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def create_prediction_table(results):
    """创建预测结果表格"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 按品种统计数据
    breed_stats = {}
    total_confidence = 0
    total_correct = 0
    total_images = len(results)
    
    for result in results:
        # 从文件名获取真实品种
        filename = os.path.basename(result['image_path'])
        true_breed = filename.split('_')[0]
        
        # 初始化品种统计
        if true_breed not in breed_stats:
            breed_stats[true_breed] = {
                'total': 0,
                'correct': 0,
                'confidence_sum': 0
            }
        
        # 更新统计
        breed_stats[true_breed]['total'] += 1
        is_correct = true_breed == result['predicted_breed']
        if is_correct:
            breed_stats[true_breed]['correct'] += 1
            total_correct += 1
        breed_stats[true_breed]['confidence_sum'] += result['confidence']
        total_confidence += result['confidence']
    
    # 准备表格数据
    table_data = []
    for breed, stats in breed_stats.items():
        accuracy = (stats['correct'] / stats['total']) * 100
        avg_confidence = stats['confidence_sum'] / stats['total']
        table_data.append([
            breed,
            f"{stats['correct']}/{stats['total']}",
            f"{accuracy:.1f}%",
            f"{avg_confidence:.1f}%"
        ])
    
    # 添加总体统计行
    total_accuracy = (total_correct / total_images) * 100
    avg_total_confidence = total_confidence / total_images
    table_data.append(['---', '---', '---', '---'])
    table_data.append([
        '总计',
        f"{total_correct}/{total_images}",
        f"{total_accuracy:.1f}%",
        f"{avg_total_confidence:.1f}%"
    ])
    
    # 创建图形
    fig_height = len(table_data) * 0.5 + 1
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=['品种', '正确/总数', '准确率', '平均置信度'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置样式
    for i in range(len(table_data)):
        if i == len(table_data) - 2:  # 分隔行
            for j in range(4):
                cell = table[i+1, j]
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f0f0f0')
        elif i == len(table_data) - 1:  # 统计行
            for j in range(4):
                cell = table[i+1, j]
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e6f3ff')
    
    # 设置标题
    plt.title('狗品种预测统计表', pad=5, fontsize=14)
    plt.subplots_adjust(top=0.95)
    
    # 保存图片
    plt.savefig('breed_statistics.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.2)
    plt.close()

def batch_predict(image_dir, model_path, data_dir):
    """批量预测文件夹中的图片"""
    # 加载数据集以获取类别信息
    dataset = DogBreedDataset(data_dir)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model = DogBreedClassifier(num_classes=len(dataset.classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 存储预测结果
    results = []
    
    # 获取所有图片文件
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"\n开始预测 {len(image_files)} 张图片...")
    
    # 批量预测
    for img_path in tqdm(image_files):
        try:
            # 加载并处理图像
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            
            # 预测
            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            # 获取预测结果
            predicted_breed = dataset.classes[predicted.item()]
            confidence_value = confidence.item() * 100
            
            # 获取所有品种的概率
            all_probs = {
                breed: prob.item() * 100 
                for breed, prob in zip(dataset.classes, probabilities[0])
            }
            
            # 存储结果
            results.append({
                'image_path': img_path,
                'predicted_breed': predicted_breed,
                'confidence': confidence_value,
                'all_probabilities': all_probs
            })
            
        except Exception as e:
            print(f"\n处理图片 {img_path} 时出错: {str(e)}")
            continue
    
    return results

def display_results(results):
    """显示预测结果"""
    print("\n预测结果汇总:")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\n图片 {i}: {os.path.basename(result['image_path'])}")
        print(f"预测品种: {result['predicted_breed']}")
        print(f"置信度: {result['confidence']:.2f}%")
        
        # 显示前3个最可能的品种
        print("前3个最可能的品种:")
        sorted_probs = sorted(
            result['all_probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        for breed, prob in sorted_probs:
            print(f"  - {breed}: {prob:.2f}%")
        print("-" * 30)
    
    # 创建可视化表格
    create_prediction_table(results)
    print("\n预测结果表已保存为 'prediction_results_table.png'")

if __name__ == "__main__":
    # 使用示例
    results = batch_predict(
        image_dir="images",    # 您的图片文件夹路径
        model_path="models/best_model.pth",  # 您的模型文路径
        data_dir="data/"                     # 训练数据目录
    )
    
    # 显示结果
    display_results(results)