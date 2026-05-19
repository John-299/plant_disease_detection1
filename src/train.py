import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import time
import traceback
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import uuid
import shutil
import json
import copy


# 1. 日志配置
def setup_logging(log_dir: str) -> logging.Logger:
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('model_training')
    logger.setLevel(logging.DEBUG)

    # 文件日志 (最大10MB，保留3个备份)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'training.log'),
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # 控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s - %(message)s'
    ))

    # 避免重复添加 handler
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


# 2. 配置类
class TrainingConfig:
    """训练配置参数"""

    def __init__(self):
        # 数据路径
        self.DATA_DIR = os.path.join('..', 'data', '猕猴桃')

        # 模型保存路径
        self.MODEL_DIR = os.path.join('..', 'models')
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        # 日志路径
        self.LOG_DIR = os.path.join('..', 'logs')
        os.makedirs(self.LOG_DIR, exist_ok=True)

        # 训练参数
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 25
        self.LEARNING_RATE = 0.001
        self.VAL_SPLIT = 0.2  # 验证集比例

        # 图像预处理参数
        self.NORMALIZE_MEAN = [0.485, 0.456, 0.406]
        self.NORMALIZE_STD = [0.229, 0.224, 0.225]

        # 预处理尺寸
        self.TRAIN_SIZE = (256, 256)
        self.VAL_SIZE = (224, 224)
        self.CROP_SIZE = 224

        # 初始化日志
        self.logger = setup_logging(self.LOG_DIR)

        # 类别名称
        self.class_names = []  # type: list[str]

        # 模型元数据
        self.model_metadata = {
            'model_version': f'1.0.{int(time.time())}',
            'model_uuid': str(uuid.uuid4()),
            'trained_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'normalize_mean': self.NORMALIZE_MEAN,
            'normalize_std': self.NORMALIZE_STD,
            'train_preprocess': {
                'resize': self.TRAIN_SIZE,
                'crop': self.CROP_SIZE,
                'augmentations': ['random_rotation', 'random_hflip', 'random_vflip', 'color_jitter',
                                  'random_resized_crop']
            },
            'val_preprocess': {
                'resize': self.VAL_SIZE,
                'crop': self.VAL_SIZE[0],
                'augmentations': []
            }
        }


# 3. 数据集加载与预处理
def prepare_datasets(config: TrainingConfig):
    """准备训练和验证数据集"""
    try:
        # 检查数据路径
        if not os.path.exists(config.DATA_DIR):
            raise FileNotFoundError(f"数据目录不存在: {config.DATA_DIR}")

        # 获取类别名称（与文件夹名称一致）
        config.class_names = sorted(os.listdir(config.DATA_DIR))
        config.model_metadata['class_names'] = config.class_names
        config.model_metadata['num_classes'] = len(config.class_names)

        config.logger.info(f"找到 {len(config.class_names)} 个类别: {config.class_names}")
        config.logger.info(f"训练预处理: {config.model_metadata['train_preprocess']}")
        config.logger.info(f"验证预处理: {config.model_metadata['val_preprocess']}")

        # 数据预处理 - 训练集
        train_transform = transforms.Compose([
            transforms.Resize(config.TRAIN_SIZE),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(config.CROP_SIZE, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ])

        # 数据预处理 - 验证集
        val_transform = transforms.Compose([
            transforms.Resize(config.VAL_SIZE),
            transforms.CenterCrop(config.CROP_SIZE),  # 添加中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ])

        # 加载完整数据集 - 使用训练转换
        full_dataset = datasets.ImageFolder(config.DATA_DIR, transform=train_transform)

        # 分割训练集和验证集
        val_size = int(config.VAL_SPLIT * len(full_dataset))
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # 确保可复现
        )

        # 为验证集创建独立副本并应用验证转换
        # 避免修改原始数据集
        val_dataset = copy.deepcopy(val_dataset)
        val_dataset.dataset.transform = val_transform  # type: ignore

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        config.logger.info(f"数据集大小: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}")
        return train_loader, val_loader

    except Exception as e:
        config.logger.error(f"数据集准备失败: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


# 4. 模型初始化
def create_model(config: TrainingConfig):
    """创建并初始化模型"""
    try:
        # 使用MobileNetV2作为基础模型
        try:
            # 兼容不同版本的PyTorch
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        except AttributeError:
            # 旧版PyTorch
            model = models.mobilenet_v2(pretrained=True)

        # 冻结所有卷积层
        for param in model.parameters():
            param.requires_grad = False

        # 修改最后一层以适应类别数量
        in_features = getattr(model.classifier[1], 'in_features', None)
        if not isinstance(in_features, int):
            raise ValueError("无法获取 classifier[1] 的 in_features")
        model.classifier[1] = nn.Linear(in_features, config.model_metadata['num_classes'])

        # 记录模型架构信息
        config.model_metadata['model_arch'] = 'mobilenet_v2'
        config.model_metadata['classifier_input'] = in_features

        config.logger.info(f"模型初始化完成 (架构: {config.model_metadata['model_arch']})")
        config.logger.info(f"分类器输入特征数: {in_features}, 输出类别数: {config.model_metadata['num_classes']}")

        return model

    except Exception as e:
        config.logger.error(f"模型创建失败: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


# 5. 训练函数
def train_model(model, train_loader, val_loader, config: TrainingConfig):
    """训练模型并验证"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        # 使用验证准确率作为调度指标
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # 明确指定监控准确率
            factor=0.1,
            patience=3,
            verbose=True
        )

        # 记录最佳准确率
        best_val_acc = 0.0

        config.logger.info("开始训练...")
        config.logger.info(f"使用设备: {device}")

        for epoch in range(config.NUM_EPOCHS):
            epoch_start = time.time()

            # 训练阶段
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()

            train_acc = 100. * correct_train / total_train
            avg_train_loss = train_loss / len(train_loader)

            # 验证阶段
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += predicted.eq(labels).sum().item()

            val_acc = 100. * correct_val / total_val
            avg_val_loss = val_loss / len(val_loader)

            # 学习率调整 - 使用验证准确率
            scheduler.step(val_acc)

            # 记录训练进度
            epoch_time = time.time() - epoch_start
            config.logger.info(
                f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}] | "
                f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model, config, is_best=True)
                config.logger.info(f"🌟 新的最佳模型! 验证准确率: {best_val_acc:.2f}%")

        config.logger.info(f"训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
        return model

    except Exception as e:
        config.logger.error(f"训练过程中出错: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


# 6. 模型保存函数
def save_model(model, config: TrainingConfig, is_best=False, is_final=False):
    """保存模型及相关元数据"""
    try:
        # 模型检查点
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_arch': config.model_metadata['model_arch'],
            'classifier_input': config.model_metadata['classifier_input'],
            'num_classes': config.model_metadata['num_classes'],
            'normalize_mean': config.model_metadata['normalize_mean'],
            'normalize_std': config.model_metadata['normalize_std'],
            'class_names': config.model_metadata['class_names'],
            'model_version': config.model_metadata['model_version'],
            'model_uuid': config.model_metadata['model_uuid'],
            'trained_date': config.model_metadata['trained_date'],
            'preprocess_metadata': {  # 保存完整的预处理元数据
                'train_preprocess': config.model_metadata['train_preprocess'],
                'val_preprocess': config.model_metadata['val_preprocess']
            }
        }

        # 基础文件名
        base_filename = 'leaf_model'

        # 保存最佳模型
        if is_best:
            model_path = os.path.join(config.MODEL_DIR, f"{base_filename}.pth")
            torch.save(checkpoint, model_path)
            config.logger.info(f"✅ 保存最佳模型到: {model_path}")

            # 同时保存元数据到JSON文件
            metadata_path = os.path.join(config.MODEL_DIR, f"{base_filename}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(config.model_metadata, f, indent=4)

            # 创建版本化备份
            versioned_path = os.path.join(
                config.MODEL_DIR,
                f"{base_filename}_v{config.model_metadata['model_version']}.pth"
            )
            shutil.copyfile(model_path, versioned_path)
            config.logger.info(f"✅ 创建版本备份: {versioned_path}")

        # 保存最终模型 (仅在最终调用时保存)
        if is_final:
            final_path = os.path.join(config.MODEL_DIR, f"{base_filename}_final.pth")
            torch.save(checkpoint, final_path)
            config.logger.info(f"✅ 保存最终模型到: {final_path}")

    except Exception as e:
        config.logger.error(f"模型保存失败: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


# 7. 主程序
def main():
    """主训练流程"""
    config = None
    try:
        # 初始化配置
        config = TrainingConfig()
        config.logger.info("🚀🚀 启动叶片病虫害模型训练")
        config.logger.info(f"模型版本: {config.model_metadata['model_version']}")
        config.logger.info(f"模型UUID: {config.model_metadata['model_uuid']}")

        # 准备数据集
        train_loader, val_loader = prepare_datasets(config)

        # 创建模型
        model = create_model(config)

        # 训练模型
        trained_model = train_model(model, train_loader, val_loader, config)

        # 保存最终模型 (单独调用)
        save_model(trained_model, config, is_final=True)

        config.logger.info("✅ 训练流程完成")

    except KeyboardInterrupt:
        if config:
            config.logger.info("用户中断训练")
    except Exception as e:
        if config:
            config.logger.error(f"主程序出错: {str(e)}\n{traceback.format_exc()}")
        else:
            print(f"主程序出错: {str(e)}")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()