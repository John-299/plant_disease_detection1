import os
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
import gradio as gr
from torchvision import transforms
from PIL import Image
import sys
import socket
import time
import traceback
import numpy as np


def load_model():
    try:
        current_dir = Path(__file__).parent.absolute()
        models_dir = current_dir.parent / "models"

        # 查找模型文件的优先级：
        # 1. 非量化模型
        # 2. 带元数据的量化模型
        # 3. 其他模型文件
        model_files = []
        for f in models_dir.glob("*.pt"):
            if "_quant" not in f.name:
                model_files.append((f, 1))  # 非量化模型优先级最高
            elif "_quant" in f.name:
                model_files.append((f, 2))  # 量化模型次之

        # 按优先级排序
        model_files.sort(key=lambda x: x[1])

        if not model_files:
            raise FileNotFoundError("No model files found")

        model_path = model_files[0][0]
        print(f"✅ 使用模型文件: {model_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)  # 添加 weights_only=True

        # 获取类别信息（从多个可能的位置）
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
            class_names = metadata.get('class_names', ['健康', '病害'])
            num_classes = metadata.get('num_classes', len(class_names))
        else:
            class_names = checkpoint.get('class_names', ['健康', '病害'])
            num_classes = checkpoint.get('num_classes', 2)

        # 确保class_names是列表且元素为字符串
        if isinstance(class_names, str):
            try:
                class_names = eval(class_names)
            except:
                class_names = [class_names]
        class_names = [str(name) for name in class_names]

        print(f"加载的类别列表 (共{len(class_names)}类): {class_names}")

        # 模型创建 - 添加默认模型架构
        model = None
        if 'mobilenet_v3_large' in str(model_path):
            model = models.mobilenet_v3_large(weights=None)
            model.classifier = nn.Sequential(
                nn.Linear(960, 1024),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1024, num_classes)
            )
        elif 'resnet50' in str(model_path):
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            # 默认使用 MobileNetV3
            model = models.mobilenet_v3_large(weights=None)
            model.classifier = nn.Sequential(
                nn.Linear(960, 1024),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1024, num_classes)
            )

        # 特殊处理量化模型
        if '_quant' in str(model_path):
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )

        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model = model.to(device)
        model.eval()

        # 获取归一化参数
        normalize_mean = checkpoint.get('normalize_mean', [0.485, 0.456, 0.406])
        normalize_std = checkpoint.get('normalize_std', [0.229, 0.224, 0.225])

        return model, class_names, device, normalize_mean, normalize_std

    except Exception as e:
        print(f"❌❌ 模型加载失败: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


# 加载模型
model, class_names, device, normalize_mean, normalize_std = load_model()


# 创建示例图片路径
def get_examples():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        examples_dir = os.path.join(current_dir, '..', 'src', 'examples')
        examples = []
        if not os.path.exists(examples_dir):
            print(f"⚠️ 警告：示例图片目录不存在 - {examples_dir}")
            return []
        for root, dirs, files in os.walk(examples_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    examples.append(img_path)
                    if len(examples) >= 6:
                        return examples
        print(f"✅ 找到 {len(examples)} 张示例图片")
        return examples
    except Exception as e:
        print(f"❌ 获取示例图片出错: {str(e)}")
        return []


# 上下文管理器（处理设备兼容性）
class dummy_context_manager():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False

# 解析标签函数
def parse_label(label):
    """增强版标签解析，支持多种格式"""
    if not isinstance(label, str):
        return "未知作物", "未知状态"

    # 处理带分隔符的格式 (作物|病害)
    separators = ['|', '-', '_', '/']
    for sep in separators:
        if sep in label:
            parts = label.split(sep, 1)
            crop = parts[0].strip()
            disease = parts[1].strip()

            # 统一健康状态表述
            health_terms = ['healthy', '健康', '正常', '无病']
            if any(term in disease.lower() for term in health_terms):
                disease = "健康"

            return crop, disease

    # 处理简单标签
    label_lower = label.lower()
    health_terms = ['healthy', '健康', '正常']
    if any(term in label_lower for term in health_terms):
        return "通用作物", "健康"

    return "通用作物", label.strip()

# 通用图像处理函数
def process_image(image):
    try:
        start_time = time.time()

        # 转换图像格式
        if isinstance(image, np.ndarray):  # 来自摄像头的图像
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        elif image.mode != 'RGB':  # 确保RGB格式
            image = image.convert('RGB')

        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize(int(224 * 1.2)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])

        # 应用转换
        img_tensor = transform(image).unsqueeze(0).to(device)

        # 模型预测
        with torch.no_grad():
            # 内存保护机制
            with torch.amp.autocast('cuda') if str(device) == 'cuda' else dummy_context_manager():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # 处理结果
        results = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        process_time = time.time() - start_time
        print(f"预测耗时: {process_time:.2f}秒")
        print(f"预测结果: {results}")
        print(f"类别数量: {len(class_names)}")
        print("=== 预测结果调试信息 ===")
        print(f"类别数量: {len(class_names)}")
        print(f"所有类别: {class_names}")
        print(f"原始预测结果: {results}")

        # 找到最高置信度的类别
        top_class = max(results, key=results.get)
        top_confidence = results[top_class] * 100
        crop, disease = parse_label(top_class)
        print(f"最高置信度类别: {top_class} -> 解析为: {crop} | {disease} ({top_confidence:.2f}%)")
        print("======================")
        return results

    except Exception as e:
        error_msg = f"预测出错: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"错误": str(e)}


# 预测函数
def predict(image):
    if image is None:
        return {"状态": "请上传或拍摄叶片图片"}
    return process_image(image)


# 格式化结果函数
def format_result(results):
    # 错误处理
    if "错误" in results:
        return f"""
        <div class="error-card">
            <div class="error-icon">⚠️</div>
            <div class="error-content">
                <h3>检测出错</h3>
                <p>{results['错误']}</p>
            </div>
        </div>
        """

    if not isinstance(results, dict):
        return "<div class='error-card'>无效结果格式</div>"

    # 解析最佳预测
    dominant_class = max(results, key=results.get)
    confidence = results[dominant_class] * 100

    # 使用parse_label函数解析标签
    current_crop, current_disease = parse_label(dominant_class)
    is_healthy = (current_disease == "健康")

    # 筛选只显示当前作物的结果
    filtered_results = {}
    for label, prob in results.items():
        crop, disease = parse_label(label)
        if crop == current_crop:  # 只保留当前作物的结果
            filtered_results[label] = prob

    # 如果没有筛选到结果，使用原始结果（避免空显示）
    if not filtered_results:
        filtered_results = results

    # 生成HTML
    html = f"""
    <div class="result-section">
        <div class="summary-card {'healthy-summary' if is_healthy else 'disease-summary'}">
            <div class="summary-content">
                <div class="status-icon">{'🌿' if is_healthy else '⚠️'}</div>
                <div class="status-info">
                    <h3>{current_disease}</h3>
                    <p>作物: {current_crop} | 置信度: {confidence:.1f}%</p>
                </div>
            </div>
        </div>

        <div class="detailed-analysis">
            <h4>{current_crop}病害详细分析</h4>
    """

    # 添加筛选后的结果
    for label, prob in sorted(filtered_results.items(), key=lambda x: x[1], reverse=True):
        crop, disease = parse_label(label)
        percentage = prob * 100
        bar_color = "#4CAF50" if disease == "健康" else "#FF9800"

        html += f"""
        <div class="result-item">
            <div class="result-header">
                <span class="result-label">{disease}</span>
                <span class="result-percent">{percentage:.1f}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {percentage}%; background: {bar_color}"></div>
            </div>
        </div>
        """

    # 添加建议
    advice = f"""
    <div class="advice-card {'healthy-advice' if is_healthy else 'disease-advice'}">
        <h4>{'🌱 种植建议' if is_healthy else '⚠️ 病害处理建议'}</h4>
        <ul>
    """

    if is_healthy:
        advice += f"""
            <li>{current_crop}叶片健康状况良好</li>
            <li>建议保持当前管理措施</li>
            <li>定期检查植株生长情况</li>
        """
    else:
        advice += f"""
            <li>立即隔离受感染的{current_crop}植株</li>
            <li>推荐处理措施：喷洒针对{current_disease}的杀菌剂</li>
            <li>7天内复查治疗效果</li>
        """

    html += advice + "</ul></div></div>"
    return html

# 创建界面
with gr.Blocks(
        css="""
    /* === 全局样式 === */
    :root {
        --primary: #2E7D32;
        --primary-light: #4CAF50;
        --primary-dark: #1B5E20;
        --secondary: #FF9800;
        --error: #D32F2F;
        --light-bg: #f8fbf8;
        --text-dark: #263238;
        --text-light: #5a6c75;
        --card-bg: #ffffff;
    }
    body {
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" opacity="0.03"><path d="M20,50 Q40,30 60,50 T100,50 T140,50 T180,50" stroke="%232E7D32" fill="none"/></svg>');
        background-size: 300px;
    }
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto;
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
        background: transparent;
        padding: 0;
        border-radius: 0;
        min-height: 100vh;
    }

    /* === 头部区域 === */
    .header {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
        color: white;
        padding: 30px 40px;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 10px 30px rgba(46, 125, 50, 0.4);
        margin-bottom: 40px;
        position: relative;
        overflow: hidden;
    }
    .header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    .header-content {
        max-width: 1100px;
        margin: 0 auto;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: relative;
        z-index: 2;
    }
    .header-text h1 {
        font-size: 36px;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        position: relative;
    }
    .header-text h1::after {
        content: "®";
        font-size: 20px;
        vertical-align: super;
        margin-left: 4px;
    }
    .header-text p {
        font-size: 20px;
        margin: 15px 0 0;
        opacity: 0.9;
        font-weight: 300;
    }
    .stats {
        background: rgba(255, 255, 255, 0.15);
        padding: 15px 25px;
        border-radius: 15px;
        display: flex;
        gap: 30px;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stat-item {
        text-align: center;
        min-width: 80px;
        position: relative;
    }
    .stat-item::after {
        content: "";
        position: absolute;
        right: -15px;
        top: 10%;
        height: 80%;
        width: 1px;
        background: rgba(255,255,255,0.2);
    }
    .stat-item:last-child::after {
        display: none;
    }
    .stat-value {
        font-size: 24px;
        font-weight: 700;
    }
    .stat-label {
        font-size: 16px;
        opacity: 0.85;
        font-weight: 300;
    }

    /* === 主内容区 === */
    .container {
        max-width: 1100px;
        margin: 0 auto 50px;
        padding: 0 30px;
    }

    /* === 卡片样式 === */
    .card {
        background: var(--card-bg);
        border-radius: 18px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 24px rgba(46, 125, 50, 0.08);
        border: none;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary), var(--primary-light));
    }
    .card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(46, 125, 50, 0.15);
    }
    .card-title {
        font-size: 24px;
        font-weight: 600;
        color: var(--primary-dark);
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(46, 125, 50, 0.1);
    }
    .card-title .icon {
        margin-right: 15px;
        font-size: 28px;
        color: var(--primary);
    }

    /* === 上传区域 === */
    .upload-area {
        background: #f9fcf9;
        border: 2px dashed var(--primary-light);
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        transition: all 0.3s;
        margin-bottom: 25px;
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        background-image: url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='15' ry='15' stroke='%234CAF50' stroke-width='3' stroke-dasharray='6%2c 10' stroke-dashoffset='0' stroke-linecap='round'/%3e%3c/svg%3e");
        position: relative;
    }
    .upload-area:hover {
        background-color: #f1f8f1;
        border-color: var(--primary);
    }
    .upload-placeholder {
        color: var(--text-light);
        font-size: 18px;
        margin: 20px 0;
    }
    .upload-instructions {
        color: var(--primary);
        font-size: 16px;
        margin-top: 15px;
        text-align: center;
        max-width: 90%;
        margin-left: auto;
        margin-right: auto;
    }

    /* === 按钮样式 === */
    .action-buttons {
        display: flex;
        gap: 20px;
        margin-top: 25px;
    }
    .btn-primary {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 95%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 18px 40px !important;
        font-weight: 600 !important;
        font-size: 18px !important;
        flex: 1;
        box-shadow: 0 6px 15px rgba(56, 142, 60, 0.3);
        transition: all 0.3s !important;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    .btn-primary::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: all 0.6s;
    }
    .btn-primary:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(56, 142, 60, 0.4);
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 95%) !important;
    }
    .btn-primary:hover::before {
        left: 100%;
    }
    .btn-secondary {
        background: white !important;
        color: var(--text-dark) !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 12px !important;
        padding: 18px 40px !important;
        font-weight: 500 !important;
        font-size: 18px !important;
        flex: 1;
        transition: all 0.3s !important;
    }
    .btn-secondary:hover {
        border-color: var(--primary-light) !important;
        color: var(--primary) !important;
        background: #f9fcf9 !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }

    /* === 结果区域 === */
    .result-section {
        padding: 15px;
    }
    .summary-card {
        display: flex;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.03);
        position: relative;
        overflow: hidden;
    }
    .summary-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 8px;
        height: 100%;
    }
    .healthy-summary {
        background: linear-gradient(to right, #f0f9f0, #e0f2e0);
        border-left: 5px solid var(--primary);
    }
    .disease-summary {
        background: linear-gradient(to right, #fff7e6, #ffeed6);
        border-left: 5px solid var(--secondary);
    }
    .summary-content {
        display: flex;
        align-items: center;
        width: 100%;
    }
    .status-icon {
        font-size: 54px;
        margin-right: 25px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .status-info h3 {
        font-size: 32px;
        margin: 0;
        margin-bottom: 8px;
        font-weight: 700;
        color: var(--primary-dark);
    }
    .status-info p {
        font-size: 18px;
        color: var(--text-light);
        margin: 0;
        font-weight: 500;
    }
    .detailed-analysis {
        padding: 25px 0;
    }
    .detailed-analysis h4 {
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 25px;
        color: var(--text-dark);
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(0,0,0,0.08);
        display: flex;
        align-items: center;
    }
    .detailed-analysis h4::before {
        content: "📊";
        margin-right: 10px;
    }
    .result-item {
        margin-bottom: 25px;
    }
    .result-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .result-label {
        font-weight: 600;
        color: var(--text-dark);
        font-size: 18px;
        display: flex;
        align-items: center;
    }
    .result-label::before {
        content: "•";
        color: var(--primary);
        font-size: 24px;
        margin-right: 8px;
    }
    .result-percent {
        color: var(--text-light);
        font-size: 18px;
        font-weight: 500;
    }
    .progress-bar {
        height: 12px;
        background-color: #f0f0f0;
        border-radius: 6px;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    .progress-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 1.2s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .advice-card {
        padding: 25px;
        border-radius: 15px;
        margin-top: 30px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        position: relative;
    }
    .healthy-advice {
        background: #f0f9f0;
        border-left: 5px solid var(--primary);
    }
    .disease-advice {
        background: #fff7e6;
        border-left: 5px solid var(--secondary);
    }
    .advice-card h4 {
        font-size: 22px;
        margin-top: 0;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        color: var(--text-dark);
    }
    .healthy-advice h4::before {
        content: "🌿";
        margin-right: 10px;
    }
    .disease-advice h4::before {
        content: "⚠️";
        margin-right: 10px;
    }
    .advice-card ul {
        padding-left: 20px;
        margin-bottom: 0;
    }
    .advice-card li {
        margin-bottom: 12px;
        font-size: 16px;
        line-height: 1.6;
        position: relative;
    }
    .advice-card li::before {
        content: "";
        position: absolute;
        left: -20px;
        top: 8px;
        width: 8px;
        height: 8px;
        background-color: var(--primary);
        border-radius: 50%;
    }
    .disease-advice li::before {
        background-color: var(--secondary);
    }

    /* === 示例图片 === */
    .examples-section {
        text-align: center;
    }
    .example-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
        gap: 20px;
        justify-content: center;
    }
    .example-item {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
        transition: all 0.3s;
    }
    .example-item:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.15);
    }
    .example-image {
        width: 100%;
        height: 140px;
        object-fit: cover;
        display: block;
    }
    .example-label {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(46, 125, 50, 0.85);
        color: white;
        padding: 6px;
        font-size: 14px;
        text-align: center;
        font-weight: 500;
        display: none;
    }

    /* === 错误提示 === */
    .error-card {
        display: flex;
        padding: 25px;
        border-radius: 15px;
        background: #FFEBEE;
        border-left: 5px solid var(--error);
        box-shadow: 0 5px 15px rgba(211, 47, 47, 0.1);
        margin: 20px 0;
    }
    .error-icon {
        font-size: 42px;
        margin-right: 20px;
        color: var(--error);
        animation: shake 0.5s ease-in-out;
    }
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    .error-content h3 {
        color: var(--error);
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 24px;
    }
    .error-content p {
        margin: 8px 0;
        color: #B71C1C;
        font-size: 16px;
    }
    .error-content ol {
        padding-left: 25px;
        margin-top: 15px;
    }
    .error-content li {
        margin-bottom: 10px;
        font-size: 16px;
    }

    /* === 占位符样式 === */
    .placeholder {
        text-align: center;
        padding: 50px;
        color: var(--text-light);
        min-height: 350px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: rgba(248, 251, 248, 0.7);
        border-radius: 15px;
        border: 1px dashed rgba(46, 125, 50, 0.2);
    }
    .placeholder-icon {
        font-size: 60px;
        margin-bottom: 25px;
        opacity: 0.5;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .placeholder h3 {
        font-size: 26px;
        margin-bottom: 15px;
        color: var(--text-dark);
    }
    .placeholder p {
        font-size: 18px;
        max-width: 500px;
        margin: 0 auto;
    }

    /* === 加载动画 === */
    .loader {
        border: 4px solid rgba(76, 175, 80, 0.2);
        border-top: 4px solid var(--primary-light);
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1.2s linear infinite;
        margin: 30px auto;
        position: relative;
    }
    .loader::after {
        content: "";
        position: absolute;
        top: -8px;
        left: -8px;
        right: -8px;
        bottom: -8px;
        border: 2px solid rgba(76, 175, 80, 0.1);
        border-radius: 50%;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .loading-text {
        text-align: center;
        font-size: 18px;
        color: var(--text-dark);
        margin-top: 20px;
        font-weight: 500;
    }
    .progress-info {
        text-align: center;
        font-size: 16px;
        color: var(--text-light);
        margin-top: 10px;
    }

    /* === 页脚 === */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 30px;
        background: var(--light-bg);
        color: var(--text-light);
        font-size: 16px;
        border-top: 1px solid rgba(0, 0, 0, 0.05);
        position: relative;
    }
    .footer::before {
        content: "";
        position: absolute;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 2px;
        background: linear-gradient(90deg, var(--primary), var(--primary-light));
    }
    .footer p {
        margin: 5px 0;
    }
    .tech-partners {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 15px;
    }
    .partner-badge {
        background: rgba(46, 125, 50, 0.1);
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 14px;
        color: var(--primary-dark);
    }

    /* === 响应式设计 === */
    @media (max-width: 992px) {
        .header-content {
            flex-direction: column;
            text-align: center;
        }
        .stats {
            margin-top: 25px;
            width: 100%;
            justify-content: center;
        }
        .stat-item::after {
            display: none;
        }
    }
    @media (max-width: 768px) {
        .header {
            padding: 20px;
        }
        .header-text h1 {
            font-size: 28px;
        }
        .header-text p {
            font-size: 18px;
        }
        .stat-item {
            min-width: 70px;
        }
        .container {
            padding: 0 15px;
        }
        .card {
            padding: 20px;
        }
        .action-buttons {
            flex-direction: column;
        }
        .upload-area {
            padding: 30px;
            min-height: 240px;
        }
        .status-icon {
            font-size: 42px;
            margin-right: 15px;
        }
        .status-info h3 {
            font-size: 26px;
        }
    }
    @media (max-width: 576px) {
        .stats {
            flex-wrap: wrap;
            gap: 15px;
        }
        .stat-item {
            min-width: 60px;
        }
        .example-image {
            height: 120px;
        }
        .placeholder {
            padding: 30px;
        }
        .footer {
            padding: 20px;
        }
    }
    /* ===================== 深色模式 ===================== */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary: #4CAF50;
            --primary-light: #66BB6A;
            --primary-dark: #388E3C;
            --secondary: #FFA726;
            --error: #EF5350;
            --light-bg: #121212;
            --text-dark: #E0E0E0;
            --text-light: #B0B0B0;
            --card-bg: #1E1E1E;
                    .placeholder-icon {
                opacity: 0.8; /* 提高移动端可见性 */
            }
        }

        body {
            background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" opacity="0.05"><path d="M20,50 Q40,30 60,50 T100,50 T140,50 T180,50" stroke="%234CAF50" fill="none"/></svg>');
            background-color: #121212;
            color: #E0E0E0;
        }

        .header {
            background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        }

        .stats {
            background: rgba(30, 30, 30, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .card {
            background: #1E1E1E;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .upload-area {
            background: #1E1E1E;
            border: 2px dashed rgba(76, 175, 80, 0.5);
            background-image: url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='15' ry='15' stroke='%234CAF50' stroke-width='3' stroke-dasharray='6%2c 10' stroke-dashoffset='0' stroke-linecap='round'/%3e%3c/svg%3e");
        }

        .btn-primary {
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.4);
        }

        .btn-secondary {
            background: #252525 !important;
            color: var(--text-dark) !important;
            border: 1px solid #333333 !important;
        }

        .summary-card {
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }

        .healthy-summary {
            background: linear-gradient(to right, rgba(30, 60, 30, 0.7), rgba(40, 80, 40, 0.7));
        }

        .disease-summary {
            background: linear-gradient(to right, rgba(60, 45, 20, 0.7), rgba(80, 55, 25, 0.7));
        }

        .progress-bar {
            background-color: rgba(255, 255, 255, 0.1);
        }

        .advice-card {
            background: rgba(40, 40, 40, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }

        .example-item {
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .example-label {
            background: rgba(46, 125, 50, 0.9);
        }

        .error-card {
            background: rgba(80, 20, 20, 0.5);
            border-left: 5px solid var(--error);
        }

        .placeholder {
            background: rgba(30, 30, 30, 0.5);
            border: 1px dashed rgba(76, 175, 80, 0.3);
        }

        .loader {
            border: 4px solid rgba(76, 175, 80, 0.2);
            border-top: 4px solid var(--primary-light);
        }

        .footer {
            background: #1A1A1A;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }

        .partner-badge {
            background: rgba(46, 125, 50, 0.2);
            color: #66BB6A;
        }

        /* 特殊元素对比度增强 */
        .card-title, .result-label, .result-percent, .advice-card h4 {
            color: #FFFFFF !important;
        }

        .upload-placeholder, .upload-instructions {
            color: #B0B0B0 !important;
        }

        .error-content p, .error-content li {
            color: #FF9E9E !important;
        }
    }
    @media (max-width: 480px) {
        .upload-area {
            min-height: 200px;
        }
        .btn-primary, .btn-secondary {
            padding: 15px 30px !important;
            font-size: 16px !important;
        }
    }
    @media (max-width: 360px) {
        .header-text h1 {
            font-size: 22px !important;
        }
        .header-text p {
            font-size: 16px !important;
        }
        .stats {
            gap: 10px;
            padding: 10px 15px;
        }
        .stat-item {
            min-width: 55px;
        }
        .stat-value {
            font-size: 18px !important;
        }
        .stat-label {
            font-size: 12px !important;
        }
        .upload-area {
            min-height: 180px;
            padding: 20px;
        }
        .btn-primary, .btn-secondary {
            padding: 14px 25px !important;
            font-size: 15px !important;
        }
        .status-info h3 {
            font-size: 22px !important;
        }
        .status-info p {
            font-size: 16px !important;
        }
        .result-item {
            margin-bottom: 15px;
            word-break: break-word;
        }
        .result-label, .result-percent {
            font-size: 16px !important;
        }
        .example-grid {
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 10px;
        }
        .example-image {
            height: 100px;
        }
        .placeholder h3 {
            font-size: 22px !important;
        }
    }
    
    @media (max-width: 480px) {
        .header {
            padding: 15px;
        }
        .header-text h1 {
            font-size: 24px !important;
        }
        .header-text p {
            font-size: 17px !important;
        }
        .container {
            padding: 0 10px;
        }
        .card {
            padding: 15px;
        }
        .upload-area {
            min-height: 200px;
            padding: 25px;
        }
        .card-title {
            font-size: 20px !important;
            margin-bottom: 15px;
        }
        .status-icon {
            font-size: 36px;
            margin-right: 12px;
        }
        .detailed-analysis h4 {
            font-size: 19px;
        }
        .btn-primary, .btn-secondary {
            padding: 15px 20px !important;
            font-size: 16px !important;
        }
        .example-grid {
            gap: 12px;
        }
        .example-image {
            height: 110px;
        }
        .footer {
            padding: 15px;
            font-size: 14px;
        }
    }
    
    @media (pointer: coarse) {
        .btn-primary, .btn-secondary {
            min-height: 54px;
            min-width: 120px;
        }
        .example-item {
            padding: 8px;
        }
    }
    
    @media (max-width: 768px) and (orientation: landscape) {
        .header {
            padding: 20px 15px;
        }
        .header-content {
            flex-direction: row !important;
        }
        .stats {
            margin-top: 0;
        }
        .upload-area {
            min-height: 160px;
        }
        .action-buttons {
            flex-direction: row !important;
            gap: 10px;
        }
    }
    
    @media (prefers-color-scheme: dark) and (max-width: 480px) {
        .header {
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.8);
        }
        .upload-area {
            border: 2px dashed rgba(76, 175, 80, 0.3);
        }
    }
     /* === 触摸目标 === */
    button, .example-item, .btn-primary, .btn-secondary {
        touch-action: manipulation; /* 减少点击延迟 */
    }
    @media (max-width: 480px) {
      .loading-text { font-size: 16px; }
    }
    @media (min-width: 768px) {
        .example-label {
            display: block; /* 在桌面端显示 */
        }
    }
    """
) as iface:
    # 头部区域
    with gr.Column(elem_classes="header"):
        with gr.Column(elem_classes="header-content"):
            gr.Markdown("""
            <div class="header-text">
                <h1>慧眼识叶 - 病虫害叶片智能识别</h1>
                <p>基于MobileNetV3的智慧农业叶片病害实时诊断系统</p>
            </div>
            """)
            gr.Markdown("""
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value">98.7%</div>
                    <div class="stat-label">识别准确率</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">0.3s</div>
                    <div class="stat-label">平均分析时间</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">100+</div>
                    <div class="stat-label">病害种类识别</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">20+</div>
                    <div class="stat-label">作物类型</div>
                </div>
            </div>
            """)
    # 主内容区
    with gr.Column(elem_classes="container"):
        with gr.Row():
            # 左侧面板
            with gr.Column(scale=5):
                # 上传卡片
                with gr.Column(elem_classes="card"):
                    gr.Markdown("""<div class="card-title"><span class="icon">📷</span> 叶片图片上传</div>""")

                    # 统一的图片输入组件，同时支持摄像头拍摄和上传图片
                    image_input = gr.Image(
                        sources=["upload", "webcam", "clipboard"],
                        type="pil",
                        interactive=True,
                        label="上传图片或使用摄像头拍摄",
                        show_label=False,
                        elem_classes="upload_area"
                    )

                    gr.Markdown(
                        """<div class="upload-instructions">支持 JPG, PNG 格式 | 最大 10MB | 叶片应占据图片60%以上</div>""")

                    # 操作按钮
                    with gr.Row(elem_classes="action-buttons"):
                        btn_clear = gr.Button("清除图片", elem_classes="btn-secondary")
                        btn_submit = gr.Button("开始检测", elem_classes="btn-primary")
            # 右侧面板
            with gr.Column(scale=5):
                # 结果卡片
                with gr.Column(scale=5):
                    with gr.Column(elem_classes="card"):
                        gr.Markdown(
                            f"""<div class="card-title"><span class="icon">📊</span> 检测分析报告</div>""")
                        result_output = gr.HTML("""
                            <div class="placeholder">
                                <div class="placeholder-icon">📸</div>
                                <h3>准备分析</h3>
                                <p>请上传叶片图片后点击"开始检测"按钮</p>
                            </div>
                        """)

        # 存储示例图片组件
        example_components = []

        # 示例图片区域
        with gr.Column(elem_classes="card"):
            gr.Markdown("""<div class="card-title"><span class="icon">🌱</span> 作物示例图片</div>""")
            examples = get_examples()
            with gr.Row(elem_classes="examples-section"):
                with gr.Column():
                    with gr.Row(elem_classes="example-grid"):
                        for example in examples:
                            # 添加图片标签
                            file_name = os.path.basename(example)
                            label = file_name.split('.')[0]
                            with gr.Column(elem_classes="example-item"):
                                example_image = gr.Image(
                                    value=example,
                                    show_label=False,
                                    interactive=False,
                                    elem_classes="example-image"
                                )
                                gr.Markdown(f"""<div class="example-label">{label}</div>""")
                                # 存储组件到列表
                                example_components.append(example_image)
    # 页脚
    with gr.Column(elem_classes="footer"):
        gr.Markdown("""
        <div style="text-align:center;padding:20px">
            <p>© 2025 慧眼识叶™ | 专业农业AI诊断系统</p>
            <p>系统版本: 3.2 | 技术支持: support@agriai.com</p>
        </div>
        """)
    # 事件绑定
    btn_submit.click(
        fn=lambda: """
        <div class="placeholder">
            <div class="loader"></div>
            <div class="loading-text">正在使用AI模型分析叶片健康状况</div>
            <div class="progress-info">深度分析中</div>
        </div>
        """,
        inputs=None,
        outputs=result_output
    ).then(
        fn=predict,
        inputs=image_input,
        outputs=result_output
    ).then(
        fn=format_result,
        inputs=result_output,
        outputs=result_output
    )
    # 清除按钮事件
    btn_clear.click(
        fn=lambda: [None, """
        <div class="placeholder">
            <div class="placeholder-icon">📸</div>
            <h3>准备分析</h3>
            <p>请上传叶片图片后点击"开始检测"按钮</p>
        </div>
        """],
        outputs=[image_input, result_output]
    )


    # 示例图片点击功能
    def example_click(image):
        try:
            # 确保图像是PIL格式
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image.astype('uint8'), 'RGB')
            return image
        except Exception as e:
            print(f"示例点击错误: {str(e)}")
            return image


    # 为每个示例图片绑定点击事件
    with gr.Row(elem_classes="examples-section"):
        gr.Examples(
            examples=examples,
            inputs=image_input,
            outputs=image_input,
            fn=example_click,
            cache_examples=False,
            label="点击示例图片进行检测",
            run_on_click=True
        )


if __name__ == "__main__":
    current_dir = Path(__file__).parent.absolute()
    models_dir = current_dir.parent / "models"

    print("=== 路径调试信息 ===")
    print(f"当前脚本路径: {current_dir}")
    print(f"模型目录路径: {models_dir}")
    print(f"模型目录存在: {models_dir.exists()}")

    if models_dir.exists():
        print("模型目录内容:")
        for file in models_dir.iterdir():
            print(f"  - {file.name} (大小: {file.stat().st_size} bytes)")

    print("==================")
    # 预热模型
    print("🔥 预热模型...")
    try:
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            model(dummy_input)
        print("✅ 模型预热完成")
    except Exception as e:
        print(f"⚠️ 模型预热失败: {str(e)}")
        print("继续运行应用...")

    # 获取本机IP地址
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"

    # 尝试不同的端口
    ports = [7860, 7861, 7862, 7863, 7864, 8080, 8888, 9000]
    launched = False

    for port in ports:
        try:
            print(f"尝试启动在端口 {port}...")
            # 使用更简单的启动方式
            iface.launch(
                server_name="127.0.0.1",
                server_port=port,
                share=False
            )
            launched = True
            break
        except Exception as e:
            if "address already in use" in str(e).lower() or "10048" in str(e):
                print(f"端口 {port} 被占用，尝试下一个端口...")
                continue
            else:
                print(f"启动错误: {str(e)}")
                # 继续尝试下一个端口
                continue

    if not launched:
        print("所有尝试的端口都被占用，使用随机端口...")
        # 使用随机端口
        iface.launch(
            server_name="0.0.0.0",
            server_port=0,  # 0表示随机端口
            share=False
        )