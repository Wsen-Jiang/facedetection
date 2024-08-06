import cv2
import numpy as np
import onnxruntime

# 打印可用的执行提供者（来自 onnxruntime）
print("Available providers:", onnxruntime.get_available_providers())

# 加载 ONNX 模型
model_path = r'C:\Users\Shawn\.insightface\models\buffalo_sc\det_500m.onnx'
session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# 使用 cv2 加载图像
img_path = 'C:/Users/Shawn/PycharmProjects/facedetection/JZD.png'
print(f"Loading image from {img_path}")
img = cv2.imread(img_path)

# 确认图像已成功加载
if img is None:
    print(f"Failed to load image: {img_path}")
else:
    print(f"Image loaded successfully. Shape: {img.shape}")

    # 缩小图像大小以减少内存需求
    img_resized = cv2.resize(img, (24, 24))
    print(f"Image resized to: {img_resized.shape}")

    # 获取模型输入信息
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].type
    print(f"Model expects input with shape: {input_shape} and type: {input_type}")

    # 检查和调整输入数据格式
    img_resized = img_resized.astype(np.float32)  # 确保数据类型正确
    img_resized = img_resized / 255.0  # 归一化
    img_transposed = img_resized.transpose(2, 0, 1)  # 转换为模型期望的格式（C, H, W）
    input_data = img_transposed[np.newaxis, :]  # 添加批次维度
    print(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}")

    # 进行推理
    try:
        print("Starting inference...")
        result = session.run(None, {input_name: input_data})
        print("Inference completed. Result:", result)
    except onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument as e:
        print(f"Invalid Argument: {e}")
    except onnxruntime.capi.onnxruntime_pybind11_state.Fail as e:
        print(f"Failed: {e}")
    except Exception as e:
        print(f"Error during inference: {e}")
