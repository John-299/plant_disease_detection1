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
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from datetime import datetime
import uuid
import gc
from PIL import Image, UnidentifiedImageError
import json
import numpy as np
import copy
import io
from collections import defaultdict
import torch.backends.cudnn as cudnn


# 增强型日志配置
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('PlantDiseaseModel')
    logger.setLevel(logging.INFO)

    # 清除现有处理器避免重复日志
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件处理器 - 带滚动备份
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'training.log'),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,  # 保留5个备份
        encoding='utf-8'
    )
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 控制台处理器 - 简洁格式
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 添加错误文件处理器
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, 'errors.log'),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)

    return logger


# 训练配置
class TrainingConfig:
    def __init__(self):
        # 目录设置
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        self.DATA_ROOT = os.path.join(project_root, 'data')
        self.MODEL_DIR = os.path.join(project_root, 'models')
        self.LOG_DIR = os.path.join(project_root, 'logs')
        self.EXPORT_DIR = os.path.join(project_root, 'export')

        # 创建目录
        for path in [self.DATA_ROOT, self.MODEL_DIR, self.LOG_DIR, self.EXPORT_DIR]:
            os.makedirs(path, exist_ok=True)

        # 日志初始化
        self.logger = setup_logger(self.LOG_DIR)
        self.logger.info(f"Data directory: {self.DATA_ROOT}")
        self.logger.info(f"Log directory: {self.LOG_DIR}")
        self.logger.info(f"Model directory: {self.MODEL_DIR}")

        # 训练参数
        self.BATCH_SIZE = 64
        self.NUM_EPOCHS = 30
        self.LEARNING_RATE = 0.001
        self.VAL_SPLIT = 0.15
        self.TEST_SPLIT = 0.15
        self.EARLY_STOP_PATIENCE = 7
        self.MODEL_ARCH = 'mobilenet_v3_large'
        self.NORMALIZE_MEAN = [0.485, 0.456, 0.406]
        self.NORMALIZE_STD = [0.229, 0.224, 0.225]
        self.INPUT_SIZE = 224
        self.USE_MIXED_PRECISION = True
        self.UNFREEZE_AFTER_EPOCH = 10
        self.SAVE_BEST_ONLY = True
        self.USE_GRADIENT_ACCUMULATION = True
        self.ACCUMULATION_STEPS = 4
        self.AUTOTUNE_BATCH_SIZE = True
        self.CHECKPOINT_PATH = os.path.join(self.MODEL_DIR, 'checkpoint.pt')

        # 模型元数据
        self.class_info = {}
        self.model_metadata = {
            'model_version': f'3.0.{int(time.time())}',
            'model_uuid': str(uuid.uuid4()),
            'trained_date': "",
            'normalize_mean': self.NORMALIZE_MEAN,
            'normalize_std': self.NORMALIZE_STD,
            'input_size': self.INPUT_SIZE,
            'class_info': {},
            'training_config': {
                'batch_size': self.BATCH_SIZE,
                'epochs': self.NUM_EPOCHS,
                'learning_rate': self.LEARNING_RATE
            },
            'model_arch': self.MODEL_ARCH
        }

        # 训练状态
        self.best_model_weights = None
        self.best_epoch = 0
        self.best_val_acc = 0.0
        self.start_time = time.time()


# 数据集分析
def analyze_dataset(root_dir, logger):
    class_info = defaultdict(lambda: {'count': 0, 'samples': [], 'crop': '', 'disease': ''})
    class_names = []  # 按索引排序的类别名称列表
    class_to_idx = {}  # 类别名称到索引的字典

    idx = 0
    for crop_dir in os.scandir(root_dir):
        if crop_dir.is_dir():
            crop_name = crop_dir.name
            for disease_dir in os.scandir(crop_dir.path):
                if disease_dir.is_dir():
                    disease_name = disease_dir.name
                    full_name = f"{crop_name}|{disease_name}"  # 使用 | 作为分隔符

                    # 添加到类别系统
                    class_names.append(full_name)
                    class_to_idx[full_name] = idx
                    idx += 1

                    # 统计图像文件
                    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
                    image_files = [
                        entry.path for entry in os.scandir(disease_dir.path)
                        if entry.is_file() and entry.name.lower().endswith(valid_extensions)
                    ]

                    # 只添加有图像的类别
                    if image_files:
                        class_info[full_name] = {
                            'count': len(image_files),
                            'samples': image_files[:3],
                            'crop': crop_name,
                            'disease': disease_name
                        }
                        logger.debug(f"Class {full_name}: {len(image_files)} images")
                    else:
                        logger.warning(f"No images found in {disease_dir.path}")

    if not class_info:
        logger.error(f"No valid data found in directory: {root_dir}")
        raise ValueError("No valid data found")

    logger.info(f"Dataset analysis complete. Found {len(class_info)} classes.")
    return dict(class_info), class_names, class_to_idx


# 动态数据集
class DynamicPlantDataset(Dataset):
    def __init__(self, root_dir, transform=None, config=None):
        self.root_dir = root_dir
        self.transform = transform
        self.config = config
        self.classes = []
        self.class_to_idx = {}
        self.samples = []
        self._cache = {}
        self.reported_errors = set()
        self._discover_classes()

        config.logger.info(f"Dataset initialized with {len(self)} samples across {len(self.classes)} classes")

    def _discover_classes(self):
        idx_counter = 0
        for crop_dir in os.scandir(self.root_dir):
            if crop_dir.is_dir():
                crop_name = crop_dir.name
                for disease_dir in os.scandir(crop_dir.path):
                    if disease_dir.is_dir():
                        class_name = f"{crop_name}|{disease_dir.name}"

                        # 添加到类别列表
                        if class_name not in self.class_to_idx:
                            self.classes.append(class_name)
                            self.class_to_idx[class_name] = idx_counter
                            idx_counter += 1

                        # 添加样本
                        class_idx = self.class_to_idx[class_name]
                        for entry in os.scandir(disease_dir.path):
                            if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                                self.samples.append((entry.path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        img_path, label = self.samples[idx]
        try:
            # 尝试直接加载
            with Image.open(img_path) as img:
                img = img.convert('RGB')
        except (IOError, UnidentifiedImageError, OSError) as e:
            try:
                # 二次加载尝试
                with open(img_path, 'rb') as f:
                    img = Image.open(io.BytesIO(f.read())).convert('RGB')
            except Exception as e2:
                if img_path not in self.reported_errors:
                    self.config.logger.error(f"Image loading error: {img_path} - {str(e2)}")
                    self.reported_errors.add(img_path)
                # 返回一个有效的占位图像和默认类别
                img = Image.new('RGB', (self.config.INPUT_SIZE, self.config.INPUT_SIZE), (128, 128, 128))
                label = 0  # 默认类别
                if self.transform:
                    img = self.transform(img)
                return img, label

        # 应用转换（如果有）
        if self.transform:
            img = self.transform(img)

        # 缓存并返回
        self._cache[idx] = (img, label)
        return img, label


# 数据增强
class SmartAugmentation:
    @staticmethod
    def get_train_transform(input_size, mean, std):
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    @staticmethod
    def get_val_transform(input_size, mean, std):
        return transforms.Compose([
            transforms.Resize(int(input_size * 1.2)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


# 模型工厂
class ModelFactory:
    @staticmethod
    def create_model(model_name, num_classes, logger):
        model_creators = {
            'resnet50': models.resnet50,
            'efficientnet_b0': models.efficientnet_b0,
            'mobilenet_v3_large': models.mobilenet_v3_large,
            'efficientnet_v2_s': models.efficientnet_v2_s
        }

        if model_name not in model_creators:
            raise ValueError(f"Unsupported model: {model_name}")

        try:
            logger.info(f"Creating {model_name} model with pretrained weights")

            # 创建基础模型
            model_creator = model_creators.get(model_name)
            if not model_creator:
                raise ValueError(f"Unsupported model: {model_name}")

            try:
                model = model_creator(weights="DEFAULT")
            except:
                # 回退到旧版API
                model = model_creator(pretrained=True)

            # 根据不同模型架构获取特征提取器的输出维度
            if 'mobilenet' in model_name:
                in_features = model.classifier[0].in_features
            elif 'resnet' in model_name:
                in_features = model.fc.in_features
            elif 'efficientnet' in model_name:
                in_features = model.classifier[1].in_features

            # 替换分类器
            if 'resnet' in model_name:
                model.fc = nn.Sequential(
                    nn.Linear(in_features, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(1024, num_classes)
                )
                logger.info(f"ResNet classifier replaced: {in_features} -> 1024 -> {num_classes}")
            else:
                model.classifier = nn.Sequential(
                    nn.Linear(in_features, 1024),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(1024, num_classes)
                )
                logger.info(f"Model classifier replaced: {in_features} -> 1024 -> {num_classes}")


        except Exception as e:
            logger.error(f"Model creation failed: {str(e)}")
            raise

        except AttributeError:
            logger.warning(f"Using deprecated pretrained=True for {model_name}")
            try:
                model = model_creator(pretrained=True)
            except:
                model = model_creator(weights=None)
            # 获取特征提取器的输出维度
            if 'mobilenet' in model_name:
                in_features = 960  # MobileNetV3特征提取器的输出维度
            elif 'resnet' in model_name:
                in_features = model.fc.in_features
            elif 'efficientnet' in model_name:
                in_features = model.classifier[1].in_features
            # 替换分类器
            if 'resnet' in model_name:
                model.fc = nn.Sequential(
                    nn.Linear(in_features, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(1024, num_classes)

                )

            else:
                model.classifier = nn.Sequential(
                    nn.Linear(in_features, 1024),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(1024, num_classes)
                )

        # 多GPU支持
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs")

        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Model created and moved to {'cuda' if torch.cuda.is_available() else 'cpu'}")
        return model


# 数据预取器
class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        except Exception as e:
            # 跳过问题批次
            print(f"Skipping problematic batch: {str(e)}")
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.to(self.device, non_blocking=True)
            self.next_target = self.next_target.to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


# 转换子集
class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)


# 训练引擎
class TrainingEngine:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = config.logger
        self.logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        # 启用cudnn优化
        if torch.cuda.is_available():
            cudnn.benchmark = True
            self.logger.info("cuDNN benchmark enabled")
        else:
            self.logger.warning("CUDA not available, using CPU")

        if torch.cuda.is_available():
            # 启用 cuDNN
            torch.backends.cudnn.enabled = True
            # 启用 cuDNN benchmark 模式以优化卷积运算
            torch.backends.cudnn.benchmark = True

        # CPU禁用混合精度
        if self.device.type == 'cpu':
            config.USE_MIXED_PRECISION = False
            self.logger.warning("Mixed precision disabled for CPU")

    def prepare_datasets(self):
        try:
            # 分析数据集
            self.config.class_info, class_names, class_to_idx = analyze_dataset(
                self.config.DATA_ROOT, self.config.logger
            )
            num_classes = len(class_names)

            self.config.model_metadata.update({
                'class_names': class_names,
                'class_to_idx': class_to_idx,
                'num_classes': len(class_names),
                'class_info': self.config.class_info
            })

            # 更新元数据
            self.config.model_metadata.update({
                'num_classes': num_classes,
                'class_info': self.config.class_info,
                'class_names': class_names,  # 新增
                'class_to_idx': class_to_idx  # 新增
            })

            # === 修复: 创建数据集时不应用任何转换 ===
            # 创建完整数据集（不应用任何转换）
            full_dataset = DynamicPlantDataset(
                self.config.DATA_ROOT, transform=None, config=self.config)  # transform=None

            # 数据集划分
            total_size = len(full_dataset)
            val_size = int(total_size * self.config.VAL_SPLIT)
            test_size = int(total_size * self.config.TEST_SPLIT)
            train_size = total_size - val_size - test_size

            train_subset, val_subset, test_subset = random_split(
                full_dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42))

            self.logger.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")

            # 分别定义训练和验证转换
            train_transform = SmartAugmentation.get_train_transform(
                self.config.INPUT_SIZE, self.config.NORMALIZE_MEAN, self.config.NORMALIZE_STD)
            val_transform = SmartAugmentation.get_val_transform(
                self.config.INPUT_SIZE, self.config.NORMALIZE_MEAN, self.config.NORMALIZE_STD)

            # 计算训练子集类别权重
            train_class_counts = torch.zeros(len(self.config.class_info))
            for idx in train_subset.indices:
                _, label = full_dataset.samples[idx]
                train_class_counts[label] += 1

            weights = 1.0 / (train_class_counts + 1e-6)
            train_labels = [full_dataset.samples[i][1] for i in train_subset.indices]
            samples_weights = weights[torch.tensor(train_labels)]

            # 采样器
            sampler = WeightedRandomSampler(
                samples_weights, len(samples_weights), replacement=True)

            # 数据加载器配置
            cpu_count = min(os.cpu_count() or 1, 8)
            num_workers = min(4, cpu_count) if self.device.type == 'cuda' else 0
            self.logger.info(f"Using {num_workers} data loader workers")

            # 训练数据加载器 - 应用训练转换
            train_loader = DataLoader(
                TransformSubset(train_subset, train_transform),
                batch_size=self.config.BATCH_SIZE,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0
            )

            # 验证数据加载器 - 应用验证转换
            val_loader = DataLoader(
                TransformSubset(val_subset, val_transform),
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=min(2, num_workers),
                pin_memory=torch.cuda.is_available()
            )

            # 测试数据加载器 - 应用验证转换
            test_loader = DataLoader(
                TransformSubset(test_subset, val_transform),
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=min(2, num_workers),
                pin_memory=torch.cuda.is_available()
            )

            return train_loader, val_loader, test_loader

        except Exception as e:
            self.logger.error(f"Dataset preparation failed: {str(e)}")
            traceback.print_exc()
            sys.exit(1)

    def create_model(self):
        try:
            model = ModelFactory.create_model(
                self.config.MODEL_ARCH,
                self.config.model_metadata['num_classes'],
                self.config.logger
            )

            # 冻结参数
            for param in model.parameters():
                param.requires_grad = False

            # 解冻分类器
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            if 'resnet' in self.config.MODEL_ARCH:
                for param in base_model.fc.parameters():
                    param.requires_grad = True
            else:
                for param in base_model.classifier.parameters():
                    param.requires_grad = True

            # 多GPU支持
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                self.logger.info(f"Using {torch.cuda.device_count()} GPUs")

            model = model.to(self.device)
            self.logger.info(f"Model created and moved to {self.device}")
            return model

        except Exception as e:
            self.logger.error(f"Model creation failed: {str(e)}")
            traceback.print_exc()
            sys.exit(1)

    def auto_tune_batch_size(self, model):
        if not self.config.AUTOTUNE_BATCH_SIZE or self.device.type != 'cuda':
            return self.config.BATCH_SIZE

        self.logger.info("Auto-tuning batch size...")
        current_batch_size = self.config.BATCH_SIZE
        max_batch_size = current_batch_size * 4
        optimal_batch_size = current_batch_size

        model.eval()
        while current_batch_size <= max_batch_size:
            try:
                # 测试张量
                dummy_input = torch.randn(current_batch_size, 3,
                                          self.config.INPUT_SIZE,
                                          self.config.INPUT_SIZE).to(self.device)

                # 前向传播测试
                with torch.no_grad():
                    model(dummy_input)

                # 成功则增加批次大小
                optimal_batch_size = current_batch_size
                current_batch_size *= 2
                self.logger.info(f"Tested batch size {optimal_batch_size} - passed")

                # 清理内存
                del dummy_input
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    self.logger.info(f"Batch size {current_batch_size} failed, OOM detected")
                    optimal_batch_size = max(8, optimal_batch_size)
                    self.logger.info(f"Optimal batch size: {optimal_batch_size}")
                    return optimal_batch_size
                else:
                    self.logger.warning(f"Batch size test error: {str(e)}")
                    break

        optimal_batch_size = min(max_batch_size, optimal_batch_size)
        self.logger.info(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size

    def train_model(self, model, train_loader, val_loader):
        # 初始化组件
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=False
        )

        scaler = None
        if self.config.USE_MIXED_PRECISION and torch.cuda.is_available():
            if hasattr(torch.cuda.amp, 'GradScaler'):
                scaler = torch.cuda.amp.GradScaler()
            elif hasattr(torch.amp, 'GradScaler'):
                scaler = torch.amp.GradScaler(device_type="cuda")
            else:
                self.logger.warning("Mixed precision training is not available")
        else:
            scaler = None

        # EMA模型
        ema_model = copy.deepcopy(model)
        ema_decay = 0.999

        # 训练状态
        self.config.best_val_acc = 0.0
        no_improve_count = 0
        global_step = 0

        # 训练恢复
        start_epoch = 0
        if os.path.exists(self.config.CHECKPOINT_PATH):
            try:
                try:
                    checkpoint = torch.load(self.config.CHECKPOINT_PATH, weights_only=True)
                except TypeError:
                    checkpoint = torch.load(self.config.CHECKPOINT_PATH)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                self.config.best_val_acc = checkpoint.get('best_val_acc', 0.0)
                self.logger.info(
                    f"Resuming training from epoch {start_epoch}, best val acc: {self.config.best_val_acc:.2f}%")
            except Exception as e:
                self.logger.error(f"Checkpoint loading error: {str(e)}")

        self.logger.info(f"Starting training for {self.config.NUM_EPOCHS} epochs...")

        # 自动调整批大小
        if self.config.AUTOTUNE_BATCH_SIZE and torch.cuda.is_available():
            batch_size = self.auto_tune_batch_size(model)
            if batch_size != self.config.BATCH_SIZE:
                self.logger.info(f"Adjusted batch size: {self.config.BATCH_SIZE} -> {batch_size}")
                self.config.BATCH_SIZE = batch_size
                self.config.model_metadata['training_config']['batch_size'] = batch_size
                train_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=batch_size,
                    sampler=train_loader.sampler,
                    num_workers=train_loader.num_workers,
                    pin_memory=True
                )

        # 训练循环
        for epoch in range(start_epoch, self.config.NUM_EPOCHS):
            start_time = time.time()
            model.train()
            train_loss, correct_train, total_train = 0.0, 0, 0
            skipped_batches = 0

            # 学习率预热
            if epoch < 3:
                lr_scale = min(1.0, float(epoch + 1) / 3)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.config.LEARNING_RATE * lr_scale
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f"Epoch {epoch + 1}: Warmup LR: {current_lr:.6f}")

            # 数据预取
            prefetcher = DataPrefetcher(train_loader, self.device)
            images, labels = prefetcher.next()
            step = 0

            while images is not None:
                accumulation_steps = self.config.ACCUMULATION_STEPS if self.config.USE_GRADIENT_ACCUMULATION else 1

                if self.config.USE_MIXED_PRECISION and torch.cuda.is_available():
                    try:
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = model(images)
                            loss = criterion(outputs, labels) / accumulation_steps
                    except (TypeError, AttributeError):
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels) / accumulation_steps
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels) / accumulation_steps

                # NaN检测
                if torch.isnan(loss).any():
                    self.logger.error(f"NaN detected in loss at step {step}")
                    optimizer.zero_grad()
                    images, labels = prefetcher.next()
                    step += 1
                    skipped_batches += 1
                    continue

                # 反向传播和权重更新
                if scaler:  # 使用混合精度
                    scaler.scale(loss).backward()

                    # 权重更新
                    if (step + 1) % accumulation_steps == 0 or (step + 1 == len(train_loader)):
                        # 取消缩放梯度以进行裁剪
                        scaler.unscale_(optimizer)

                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        # 更新参数
                        scaler.step(optimizer)
                        scaler.update()

                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                else:  # 不使用混合精度
                    loss.backward()

                    # 权重更新
                    if (step + 1) % accumulation_steps == 0 or (step + 1 == len(train_loader)):
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        # 更新参数
                        optimizer.step()

                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                # 更新EMA模型
                with torch.no_grad():
                    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

                # 更新指标
                train_loss += loss.item() * accumulation_steps
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()

                # 下一批次
                images, labels = prefetcher.next()
                step += 1

            # 计算指标
            if step - skipped_batches > 0:
                train_loss = train_loss / (step - skipped_batches)
            else:
                train_loss = 0.0

            train_acc = 100. * correct_train / total_train if total_train > 0 else 0.0

            val_loss, val_acc = self.validate_model(model, val_loader, criterion)

            scheduler.step(val_acc)

            current_lr = optimizer.param_groups[0]['lr']

            # 日志
            epoch_time = time.time() - start_time
            elapsed_time = (time.time() - self.config.start_time) / 3600  # 小时

            log_msg = (f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS} | "
                       f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}% | "
                       f"Val: loss={val_loss:.4f}, acc={val_acc:.2f}% | "
                       f"LR: {current_lr:.6f} | "
                       f"Time: {epoch_time:.1f}s | "
                       f"Elapsed: {elapsed_time:.2f}h")

            if skipped_batches > 0:
                log_msg += f" | Skipped: {skipped_batches} batches"

            self.logger.info(log_msg)

            # 保存最佳模型
            if val_acc > self.config.best_val_acc:
                self.config.best_val_acc = val_acc
                self.config.best_epoch = epoch
                no_improve_count = 0
                if self.config.SAVE_BEST_ONLY:
                    self.config.best_model_weights = copy.deepcopy(model.state_dict())
                    self.save_model(model, is_best=True)
                self.logger.info(f"🌟 New best model! Val acc: {val_acc:.2f}%")
            else:
                no_improve_count += 1
                if no_improve_count >= self.config.EARLY_STOP_PATIENCE:
                    self.logger.info(f"Early stopping triggered (no improvement for {no_improve_count} epochs)")
                    break

            # 解冻更多层（传递优化器）
            if epoch == self.config.UNFREEZE_AFTER_EPOCH:
                self.unfreeze_more_layers(model, optimizer)

            # 保存检查点
            classifier_input = self.get_classifier_input(model)

            # 判断当前模型是否是最佳模型
            is_current_best = (val_acc > self.config.best_val_acc)

            try:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_acc': self.config.best_val_acc,
                    'metadata': self.config.model_metadata,
                    'num_classes': self.config.model_metadata['num_classes'],
                    'class_names': self.config.model_metadata['class_names'],
                    'normalize_mean': self.config.NORMALIZE_MEAN,
                    'normalize_std': self.config.NORMALIZE_STD,
                    'classifier_input': classifier_input,
                    'is_best': is_current_best
                }, self.config.CHECKPOINT_PATH, weights_only=True)
            except TypeError:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_acc': self.config.best_val_acc,
                    'metadata': self.config.model_metadata,
                    'num_classes': self.config.model_metadata['num_classes'],
                    'class_names': self.config.model_metadata['class_names'],
                    'normalize_mean': self.config.NORMALIZE_MEAN,
                    'normalize_std': self.config.NORMALIZE_STD,
                    'classifier_input': classifier_input,
                    'is_best': is_current_best
                }, self.config.CHECKPOINT_PATH)

            # 内存清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 加载最佳权重
        if self.config.best_model_weights:
            model.load_state_dict(self.config.best_model_weights)
            self.logger.info(f"Loaded best weights from epoch {self.config.best_epoch}")

        return model, ema_model

    def unfreeze_more_layers(self, model, optimizer):  # 添加optimizer参数
        self.logger.info("Unfreezing additional layers...")
        layers_unfrozen = 0

        # 处理多GPU模型
        if isinstance(model, nn.DataParallel):
            base_model = model.module
        else:
            base_model = model

        # 层选择
        if 'resnet' in self.config.MODEL_ARCH:
            layers_to_unfreeze = ['layer3', 'layer4']
        elif 'efficientnet' in self.config.MODEL_ARCH:
            layers_to_unfreeze = ['features.5', 'features.6', 'features.7']
        elif 'mobilenet' in self.config.MODEL_ARCH:
            layers_to_unfreeze = ['features.10', 'features.12', 'features.14']
        else:
            layers_to_unfreeze = []

        # 解冻层
        for name, param in base_model.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True
                layers_unfrozen += 1

        # 关键修复：在解冻后重新统计参数
        total_params = 0
        trainable_params = 0
        for param in base_model.parameters():
            total_params += 1
            if param.requires_grad:
                trainable_params += 1

        # 添加解冻层后的参数统计
        self.logger.info(
            f"Unfrozen {layers_unfrozen} additional layers | "
            f"Trainable params: {trainable_params}/{total_params} ({trainable_params / total_params:.1%})"
        )

        # 更新优化器参数组
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer.param_groups = []
        optimizer.add_param_group({
            'params': trainable_params,
            'lr': self.config.LEARNING_RATE,
            'weight_decay': 0.01
        })

        self.logger.info("Optimizer updated with unfrozen parameters")

    def validate_model(self, model, val_loader, criterion):
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        skipped_batches = 0

        with torch.no_grad():
            for images, labels in val_loader:
                try:
                    # 确保数据在正确设备上
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # 混合精度验证
                    if self.config.USE_MIXED_PRECISION and torch.cuda.is_available():
                        try:
                            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                                outputs = model(images)
                                loss = criterion(outputs, labels)
                        except (TypeError, AttributeError):
                            with torch.cuda.amp.autocast():
                                outputs = model(images)
                                loss = criterion(outputs, labels)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    # 更新指标
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += predicted.eq(labels).sum().item()
                except Exception as e:
                    self.logger.warning(f"Validation batch skipped: {str(e)}")
                    skipped_batches += 1

        # 计算指标
        val_loss /= (len(val_loader) - skipped_batches) if (len(val_loader) - skipped_batches) > 0 else 1
        val_acc = 100. * correct_val / total_val if total_val > 0 else 0.0

        if skipped_batches > 0:
            self.logger.warning(f"Skipped {skipped_batches} validation batches due to errors")

        return val_loss, val_acc

    def get_classifier_input(self, model):
        """获取分类器的输入特征数"""
        # 如果是多GPU训练，需要访问module
        if isinstance(model, nn.DataParallel):
            base_model = model.module
        else:
            base_model = model

        # MobileNetV3
        if hasattr(base_model, 'classifier') and len(base_model.classifier) > 0:
            # 找到第一个线性层
            for layer in base_model.classifier:
                if isinstance(layer, nn.Linear):
                    return layer.in_features

        else:
            return 1280

    def evaluate_model(self, model, test_loader):
        model.eval()
        correct, total = 0, 0
        skipped_batches = 0

        # 数据预取
        prefetcher = DataPrefetcher(test_loader, self.device)
        images, labels = prefetcher.next()

        with torch.no_grad():
            while images is not None:
                try:
                    # 混合精度推理
                    if self.config.USE_MIXED_PRECISION and torch.cuda.is_available():
                        try:
                            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                                outputs = model(images)
                        except (TypeError, AttributeError):
                            with torch.cuda.amp.autocast():
                                outputs = model(images)
                    else:
                        outputs = model(images)

                    # 预测
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                except Exception as e:
                    self.logger.warning(f"Test batch skipped: {str(e)}")
                    skipped_batches += 1

                # 下一批次
                images, labels = prefetcher.next()

        # 计算准确率
        overall_acc = 100. * correct / total if total > 0 else 0.0

        # 更新元数据
        self.config.model_metadata['test_accuracy'] = overall_acc
        self.config.model_metadata['test_samples'] = total
        self.logger.info(f"Test accuracy: {overall_acc:.2f}% (Samples: {total}, Skipped batches: {skipped_batches})")

        return overall_acc

    def save_model(self, model, is_best=False, is_final=False):
        try:
            model_name = f"plant_disease_model_{self.config.MODEL_ARCH}"
            version = self.config.model_metadata['model_version']

            # 保存完整模型（非量化）
            model_path = os.path.join(self.config.MODEL_DIR,
                                      f"{model_name}_final_v{version}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': self.config.best_epoch if is_best else self.config.NUM_EPOCHS,
                'best_val_acc': self.config.best_val_acc,
                'metadata': self.config.model_metadata,
                'num_classes': self.config.model_metadata['num_classes'],
                'class_names': self.config.model_metadata['class_names'],
                'normalize_mean': self.config.NORMALIZE_MEAN,
                'normalize_std': self.config.NORMALIZE_STD,
                'model_arch': self.config.MODEL_ARCH
            }, model_path)
            self.logger.info(f"Full model saved: {model_path}")

            # 量化模型（确保包含元数据）
            if is_final and self.device.type == 'cuda':
                try:
                    quantized_model = torch.quantization.quantize_dynamic(
                        model, {nn.Linear}, dtype=torch.qint8
                    )
                    quant_path = os.path.join(self.config.MODEL_DIR,
                                              f"{model_name}_final_v{version}_quant.pt")
                    torch.save({
                        'model_state_dict': quantized_model.state_dict(),
                        'metadata': self.config.model_metadata,
                        'num_classes': self.config.model_metadata['num_classes'],
                        'class_names': self.config.model_metadata['class_names'],
                        'normalize_mean': self.config.NORMALIZE_MEAN,
                        'normalize_std': self.config.NORMALIZE_STD,
                        'model_arch': self.config.MODEL_ARCH
                    }, quant_path)
                    self.logger.info(f"Quantized model with metadata saved: {quant_path}")
                except Exception as e:
                    self.logger.warning(f"Quantization failed: {str(e)}")

        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise

    # 主函数
def main():
    config = TrainingConfig()
    try:
        # 初始化
        config.model_metadata['trained_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config.logger.info("Plant disease model training started")
        config.logger.info(f"Model architecture: {config.MODEL_ARCH}")
        config.logger.info(
            f"Training parameters: {config.NUM_EPOCHS} epochs, LR={config.LEARNING_RATE}, BS={config.BATCH_SIZE}")

        # 训练引擎
        engine = TrainingEngine(config)

        # 准备数据
        train_loader, val_loader, test_loader = engine.prepare_datasets()

        # 创建模型
        model = engine.create_model()

        # 训练模型
        trained_model, ema_model = engine.train_model(model, train_loader, val_loader)

        # 评估模型
        test_acc = engine.evaluate_model(trained_model, test_loader)
        ema_test_acc = engine.evaluate_model(ema_model, test_loader)
        config.logger.info(f"EMA model test accuracy: {ema_test_acc:.2f}%")

        # 保存最终模型
        engine.save_model(trained_model, is_final=True)

        # 最终报告
        total_time = (time.time() - config.start_time) / 3600
        config.logger.info(
            f"Training complete! Best val acc: {config.best_val_acc:.2f}% | "
            f"Test acc: {test_acc:.2f}% | Total time: {total_time:.2f} hours"
        )

        # 清理
        del model, trained_model, ema_model
        torch.cuda.empty_cache()
        gc.collect()

        return 0

    except KeyboardInterrupt:
        config.logger.warning("Training interrupted by user")
        return 1

    except Exception as e:
        config.logger.critical(f"Critical error: {str(e)}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())