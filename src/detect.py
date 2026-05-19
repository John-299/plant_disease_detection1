import cv2
import torch
import torchvision.models as models
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import time
import numpy as np
import traceback
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime
from torchvision import transforms
from PIL import Image
import threading
import json
import collections
import queue


class ContextFilter(logging.Filter):
    def filter(self, record):
        record.pid = os.getpid()
        record.tid = threading.get_ident()
        return True


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('camera_detection')
    logger.setLevel(logging.DEBUG)
    logger.addFilter(ContextFilter())

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'camera.log'),
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - PID:%(pid)d TID:%(tid)d - %(name)s - %(levelname)s - %(message)s'
    ))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s - %(message)s'
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class CameraConfig:
    def __init__(self):
        # 摄像头配置
        self.camera_url = "http://192.168.157.85:8080/video"
        self.camera_type = "ip"
        self.connection_timeout = 10  # 秒
        self.max_retries = 5
        self.camera_index = 0
        self.camera_width = 1280
        self.camera_height = 720
        self.camera_fps = 30

        # 模型配置
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(current_dir, '..', 'models')
        self.model_path = os.path.join(self.model_dir, 'leaf_model.pth')
        self.metadata_path = os.path.join(self.model_dir, 'leaf_model_metadata.json')
        self.log_dir = os.path.join(current_dir, '..', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

        # 显示配置
        self.window_name = "智慧农业 - 叶片实时检测"
        self.font_scale = 1.0
        self.font_thickness = 2
        self.text_color = (255, 255, 255)
        self.healthy_color = (0, 255, 0)
        self.disease_color = (0, 0, 255)
        self.fps_color = (0, 255, 255)
        self.info_color = (255, 255, 0)
        self.device_info_color = (200, 200, 255)

        # 性能配置
        self.target_fps = 15
        self.skip_frames = 1

        # USB摄像头配置
        self.usb_detect_interval = 5  # 秒，USB摄像头检测间隔
        self.usb_max_index = 10  # 最大检测的USB摄像头索引

        # 初始化日志
        self.logger = setup_logging(self.log_dir)
        self.model_metadata = None
        self._load_metadata()

    def _load_metadata(self):
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                    self.logger.info("✅ 加载模型元数据成功")
                    self.logger.info(f"模型版本: {self.model_metadata.get('model_version', '未知')}")
                    self.logger.info(f"训练日期: {self.model_metadata.get('trained_date', '未知')}")
            else:
                self.logger.warning("⚠️ 未找到模型元数据文件，使用默认值")
                self.model_metadata = {
                    'normalize_mean': [0.485, 0.456, 0.406],
                    'normalize_std': [0.229, 0.224, 0.225],
                    'class_names': ['健康', '病害'],
                    'preprocess_metadata': {
                        'train_preprocess': {'resize': (256, 256), 'crop': 224},
                        'val_preprocess': {'resize': (224, 224)}
                    }
                }
        except Exception as e:
            self.logger.error(f"加载元数据失败: {str(e)}")
            self.model_metadata = {
                'normalize_mean': [0.485, 0.456, 0.406],
                'normalize_std': [0.229, 0.224, 0.225],
                'class_names': ['健康', '病害'],
                'preprocess_metadata': {
                    'train_preprocess': {'resize': (256, 256), 'crop': 224},
                    'val_preprocess': {'resize': (224, 224)}
                }
            }

    def detect_usb_cameras(self, max_index: int = 10) -> List[Dict]:
        """检测可用的USB摄像头"""
        usb_cameras = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # 获取摄像头基本信息
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # 尝试获取更详细的设备名称
                device_name = f"USB Camera {i}"
                try:
                    # 尝试通过属性获取设备名称
                    device_name = cap.getBackendName()
                except:
                    pass

                usb_cameras.append({
                    'index': i,
                    'name': device_name,
                    'resolution': f"{width}x{height}",
                    'fps': fps
                })
                cap.release()

        self.logger.info(f"检测到 {len(usb_cameras)} 个USB摄像头")
        return usb_cameras


class ModelLoader:
    def __init__(self, config: CameraConfig):
        self.config = config

    def load(self) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        try:
            config = self.config
            if not os.path.exists(config.model_path):
                raise FileNotFoundError(f"模型文件不存在: {config.model_path}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            config.logger.info(f"使用设备: {device}")

            checkpoint = torch.load(config.model_path, map_location=device)
            model_metadata = {
                'normalize_mean': checkpoint.get('normalize_mean', config.model_metadata['normalize_mean']),
                'normalize_std': checkpoint.get('normalize_std', config.model_metadata['normalize_std']),
                'class_names': checkpoint.get('class_names', config.model_metadata['class_names']),
                'model_arch': checkpoint.get('model_arch', 'mobilenet_v2'),
                'model_version': checkpoint.get('model_version', '未知'),
                'model_uuid': checkpoint.get('model_uuid', '未知'),
                'trained_date': checkpoint.get('trained_date', '未知'),
                'preprocess_metadata': checkpoint.get('preprocess_metadata',
                                                      config.model_metadata.get('preprocess_metadata', {}))
            }

            model_arch = model_metadata['model_arch']
            config.logger.info(f"加载模型架构: {model_arch}")

            if model_arch == 'mobilenet_v2':
                try:
                    model = models.mobilenet_v2(weights=None)
                except TypeError:
                    model = models.mobilenet_v2(pretrained=False)
                classifier_input = checkpoint['classifier_input']
                model.classifier[1] = torch.nn.Linear(classifier_input, checkpoint['num_classes'])
            elif model_arch == 'resnet50':
                try:
                    model = models.resnet50(weights=None)
                except TypeError:
                    model = models.resnet50(pretrained=False)
                model.fc = torch.nn.Linear(checkpoint['classifier_input'], checkpoint['num_classes'])
            else:
                raise ValueError(f"不支持的模型架构: {model_arch}")

            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            config.logger.info(f"✅ 模型加载成功")
            return model, model_metadata

        except Exception as e:
            self.config.logger.error(f"模型加载失败: {str(e)}")
            traceback.print_exc()
            raise


class ImageProcessor:
    def __init__(self, model: torch.nn.Module, metadata: Dict[str, Any], device: torch.device):
        self.model = model
        self.metadata = metadata
        self.device = device
        self.logger = metadata.get('logger', logging.getLogger('image_processor'))

        preprocess_meta = metadata.get('preprocess_metadata', {})
        val_preprocess = preprocess_meta.get('val_preprocess', {})
        train_preprocess = preprocess_meta.get('train_preprocess', {})

        resize_size = val_preprocess.get('resize', train_preprocess.get('resize', (224, 224)))
        crop_size = val_preprocess.get('crop', train_preprocess.get('crop', 224))

        normalize_mean = metadata.get('normalize_mean', [0.485, 0.456, 0.406])
        normalize_std = metadata.get('normalize_std', [0.229, 0.224, 0.225])

        # 使用OpenCV优化图像预处理
        self.target_size = resize_size
        self.crop_size = crop_size
        self.normalize_mean = np.array(normalize_mean, dtype=np.float32)
        self.normalize_std = np.array(normalize_std, dtype=np.float32)

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        try:
            # 预处理图像 - 使用OpenCV优化
            processed_img = cv2.resize(image, self.target_size)
            # 添加颜色空间转换：BGR to RGB
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            h, w = processed_img.shape[:2]
            # 添加边界检查的裁剪逻辑
            start_x = max(0, (w - self.crop_size) // 2)
            start_y = max(0, (h - self.crop_size) // 2)
            end_x = min(w, start_x + self.crop_size)
            end_y = min(h, start_y + self.crop_size)
            cropped_img = processed_img[start_y:end_y, start_x:end_x]

            # 转换为张量
            tensor_img = torch.from_numpy(cropped_img).permute(2, 0, 1).float() / 255.0

            # 归一化
            for t, m, s in zip(tensor_img, self.normalize_mean, self.normalize_std):
                t.sub_(m).div_(s)

            input_tensor = tensor_img.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                pred = torch.argmax(probabilities).item()
                confidence = float(probabilities[pred])

            class_name = self.metadata['class_names'][pred]
            return class_name, confidence

        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            return "错误", 0.0


class CameraProcessor:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = None
        self.skip_counter = 0
        self.usb_cameras = config.detect_usb_cameras(config.usb_max_index)
        self.current_camera_index = -1
        self.current_camera_name = "未选择"
        self.last_detect_time = time.time()
        self.camera_switch_queue = queue.Queue()

    def update_usb_cameras(self):
        """定期更新USB摄像头列表"""
        current_time = time.time()
        if current_time - self.last_detect_time > self.config.usb_detect_interval:
            self.usb_cameras = self.config.detect_usb_cameras(self.config.usb_max_index)
            self.last_detect_time = current_time
            self.config.logger.debug(f"更新USB摄像头列表，找到 {len(self.usb_cameras)} 个可用设备")

    def get_usb_cameras(self):
        """获取检测到的USB摄像头列表"""
        return self.usb_cameras

    def switch_to_camera(self, index: int, name: str):
        """切换到指定摄像头"""
        self.camera_switch_queue.put((index, name))

    def process_switch_queue(self):
        """处理摄像头切换请求"""
        if not self.camera_switch_queue.empty():
            index, name = self.camera_switch_queue.get()
            self._initialize_camera(index, name)

    def _initialize_camera(self, index: int, name: str):
        """初始化指定摄像头"""
        try:
            # 释放现有资源
            if self.cap and self.cap.isOpened():
                self.cap.release()

            config = self.config
            config.logger.info(f"尝试连接摄像头: {name} (索引: {index})")

            # 区分USB和IP摄像头
            if name.startswith("USB"):
                self.cap = cv2.VideoCapture(index)
                if not self.cap.isOpened():
                    config.logger.warning(f"无法打开摄像头索引 {index}")
                    return False
            else:
                # IP摄像头处理
                self.cap = cv2.VideoCapture(name)
                start_time = time.time()
                while not self.cap.isOpened():
                    if time.time() - start_time > config.connection_timeout:
                        config.logger.warning(f"IP摄像头连接超时: {name}")
                        return False
                    time.sleep(0.1)

            # 设置摄像头参数
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
                self.cap.set(cv2.CAP_PROP_FPS, config.camera_fps)
            except Exception as e:
                config.logger.warning(f"设置摄像头参数失败: {str(e)}")

            # 获取实际设置值
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            config.logger.info(
                f"摄像头初始化成功 - 名称: {name} "
                f"分辨率: {actual_width}x{actual_height}, FPS: {actual_fps:.1f}"
            )

            # 更新当前摄像头信息
            self.current_camera_index = index
            self.current_camera_name = name
            return True

        except Exception as e:
            config.logger.error(f"摄像头初始化失败: {str(e)}")
            return False

    def initialize_default(self):
        """初始化默认摄像头（IP摄像头）"""
        self._initialize_camera(0, self.config.camera_url)

    def initialize(self, camera_type=None, camera_index=None) -> cv2.VideoCapture:
        # 保留原有初始化逻辑
        # 如果指定了摄像头类型和索引，则使用
        if camera_type is not None:
            self.config.camera_type = camera_type
        if camera_index is not None:
            self.config.camera_index = camera_index

        # ...（原有初始化逻辑保持不变）...

    def process_frame(self, frame: np.ndarray, processor: ImageProcessor) -> np.ndarray:
        try:
            config = self.config
            self.skip_counter = (self.skip_counter + 1) % (config.skip_frames + 1)
            if self.skip_counter > 0:  # 跳过指定数量的帧
                return frame

            # 添加设备信息显示
            cv2.putText(frame, f"设备: {self.current_camera_name}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, config.font_scale * 0.7,
                        config.device_info_color, config.font_thickness // 2)

            # 原始预测逻辑
            class_name, confidence = processor.predict(frame)
            label = f"{class_name}: {confidence:.2f}"
            color = config.healthy_color if "健康" in class_name else config.disease_color

            cv2.putText(frame, label, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, config.font_scale,
                        color, config.font_thickness)

            # 添加USB摄像头数量提示
            if self.usb_cameras:
                usb_text = f"按 'S' 键切换摄像头 ({len(self.usb_cameras)} 个USB可用)"
                cv2.putText(frame, usb_text, (20, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, config.font_scale * 0.6,
                            config.info_color, config.font_thickness // 2)

            return frame

        except Exception as e:
            config.logger.error(f"处理帧时出错: {str(e)}")
            return frame

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.config.logger.info("摄像头资源已释放")
        self.cap = None


class PerformanceMonitor:
    def __init__(self, max_samples=1000):
        self.prev_time = time.time()
        self.fps = 0.0
        self.last_log_time = time.time()
        # 使用双端队列限制历史数据大小
        self.processing_times = collections.deque(maxlen=max_samples)

    def update(self) -> float:
        curr_time = time.time()
        elapsed = curr_time - self.prev_time
        self.prev_time = curr_time
        self.processing_times.append(elapsed)
        self.fps = 0.9 * self.fps + 0.1 / elapsed if elapsed > 0 else self.fps
        return self.fps

    def get_performance_stats(self) -> dict:
        if not self.processing_times:
            return {}

        # 计算最近数据的统计信息
        avg_time = sum(self.processing_times) / len(self.processing_times)
        min_time = min(self.processing_times)
        max_time = max(self.processing_times)

        return {
            'fps': self.fps,
            'avg_processing_ms': avg_time * 1000,
            'min_processing_ms': min_time * 1000,
            'max_processing_ms': max_time * 1000
        }

    def add_fps_to_frame(self, frame: np.ndarray, config: CameraConfig) -> np.ndarray:
        fps = self.update()
        curr_time = time.time()

        # 每5秒记录一次详细性能数据
        if curr_time - self.last_log_time > 5:
            self.last_log_time = curr_time
            stats = self.get_performance_stats()
            if stats:  # 确保有统计数据
                config.logger.info(
                    f"性能统计: FPS={fps:.1f}, "
                    f"平均处理时间={stats['avg_processing_ms']:.1f}ms"
                )

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, config.font_scale * 0.7,
                    config.fps_color, config.font_thickness)
        return frame


def main():
    config = None
    camera_processor = None
    cap = None

    try:
        config = CameraConfig()
        config.logger.info("🚀🚀🚀🚀🚀🚀🚀🚀 启动叶片实时检测系统")

        # 加载模型
        model_loader = ModelLoader(config)
        model, model_metadata = model_loader.load()
        config.model_metadata = model_metadata
        config.model_metadata['logger'] = config.logger

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_processor = ImageProcessor(model, model_metadata, device)

        camera_processor = CameraProcessor(config)

        # 显示可用USB摄像头
        if camera_processor.usb_cameras:
            print("检测到以下USB摄像头:")
            for i, cam in enumerate(camera_processor.usb_cameras):
                print(
                    f"{i + 1}. {cam['name']} (索引{cam['index']}), 分辨率: {cam['resolution']}, FPS: {cam['fps']:.1f}")

            selection = input("请选择要使用的USB摄像头编号 (按Enter使用默认IP摄像头): ")
            if selection.isdigit():
                index = int(selection) - 1
                if 0 <= index < len(camera_processor.usb_cameras):
                    camera_name = camera_processor.usb_cameras[index]['name']
                    # 初始化选中的USB摄像头
                    if camera_processor._initialize_camera(
                            camera_processor.usb_cameras[index]['index'],
                            camera_name
                    ):
                        cap = camera_processor.cap

        # 如果没有选择USB摄像头，则初始化默认摄像头
        if not cap or not cap.isOpened():
            try:
                camera_processor.initialize_default()
                cap = camera_processor.cap
            except Exception as e:
                config.logger.error(f"摄像头初始化失败: {str(e)}")
                # 创建空白图像显示错误
                try:
                    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(blank, "摄像头初始化失败", (50, 360),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    cv2.imshow(config.window_name, blank)
                    cv2.waitKey(1)
                except:
                    pass
                raise

        perf_monitor = PerformanceMonitor()

        config.logger.info("✅ 系统初始化完成，开始实时检测...")
        target_delay = 1.0 / config.target_fps
        max_reconnect_attempts = 5
        reconnect_attempts = 0
        last_usb_detect = time.time()
        current_camera_index = 0

        while True:
            loop_start = time.time()

            # 定期更新USB摄像头列表
            if time.time() - last_usb_detect > config.usb_detect_interval:
                camera_processor.update_usb_cameras()
                last_usb_detect = time.time()

            # 处理摄像头切换请求
            camera_processor.process_switch_queue()

            # 检查摄像头是否有效
            if cap is None or not cap.isOpened():
                config.logger.warning("摄像头未初始化，尝试重新连接...")
                try:
                    cap = camera_processor.initialize()
                    if cap and cap.isOpened():
                        reconnect_attempts = 0  # 成功重连后重置计数器
                        continue  # 跳过本次循环，进入下一次
                except Exception as e:
                    config.logger.error(f"重连失败: {str(e)}")
                    reconnect_attempts += 1

            # 如果重连次数超过限制
            if reconnect_attempts >= max_reconnect_attempts:
                config.logger.error("多次重连失败，退出程序")
                break

            # 尝试读取帧
            try:
                ret, frame = cap.read()
                if not ret:
                    config.logger.warning("⚠️ 无法读取摄像头画面，尝试重新连接...")
                    reconnect_attempts += 1
                    continue
            except Exception as e:
                config.logger.error(f"读取帧失败: {str(e)}")
                reconnect_attempts += 1
                continue

            # 处理帧
            try:
                processed_frame = camera_processor.process_frame(frame, image_processor)
                processed_frame = perf_monitor.add_fps_to_frame(processed_frame, config)
                cv2.imshow(config.window_name, processed_frame)
            except Exception as e:
                config.logger.error(f"处理帧失败: {str(e)}")

            key = cv2.waitKey(1)
            if key == ord('q') or cv2.getWindowProperty(config.window_name, cv2.WND_PROP_VISIBLE) < 1:
                config.logger.info("用户请求退出")
                break
            elif key == ord('s') and camera_processor.usb_cameras:
                # 切换到下一个USB摄像头
                current_camera_index = (current_camera_index + 1) % len(camera_processor.usb_cameras)
                cam = camera_processor.usb_cameras[current_camera_index]
                camera_processor.switch_to_camera(cam['index'], cam['name'])
                config.logger.info(f"切换到摄像头: {cam['name']} (索引: {cam['index']})")

            # 帧率控制
            elapsed = time.time() - loop_start
            if elapsed < target_delay:
                time.sleep(max(0, target_delay - elapsed))

        config.logger.info("✅ 检测流程正常结束")

    except KeyboardInterrupt:
        if config: config.logger.info("用户中断操作")
    except Exception as e:
        if config:
            config.logger.error(f"主程序出错: {str(e)}")
            traceback.print_exc()
        else:
            print(f"主程序出错: {str(e)}")
            traceback.print_exc()
    finally:
        # 释放资源
        try:
            if cap and cap.isOpened():
                cap.release()
            if camera_processor:
                camera_processor.release()
        except Exception as e:
            if config:
                config.logger.error(f"释放资源时出错: {str(e)}")

        cv2.destroyAllWindows()
        if config:
            config.logger.info("系统资源已释放")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"未处理的异常: {str(e)}")
        traceback.print_exc()
        sys.exit(1)