import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
# ==========================================
# 0. 环境与 GPU 检测配置
# ==========================================
print("TensorFlow 版本:", tf.__version__)
print("是否包含 CUDA 编译支持:", tf.test.is_built_with_cuda())

# 获取物理 GPU 列表
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"✅ 成功检测到可用 GPU: {len(gpus)} 个")
    for i, gpu in enumerate(gpus):
        print(f"   - GPU {i}: {gpu.name}")
        # 设置 GPU 显存按需分配 (Memory Growth)
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"   - 已为 GPU {i} 开启显存动态增长")
        except RuntimeError as e:
            # 异常通常发生在程序启动后且物理设备已被初始化时
            print(f"   - 显存动态增长设置失败: {e}")
else:
    print("❌ 未检测到可用 GPU，TensorFlow 将默认使用 CPU 运行。")
print("=" * 42)
# ==========================================
# 1. 配置参数与路径
# ==========================================
DATA_DIR = r"E:\PycharmProjects\PythonProject1\captcha_images_v2"
TEST_SIZE = 0.1  # 10% 测试集
IMG_WIDTH = 200
IMG_HEIGHT = 50
BATCH_SIZE = 16
EPOCHS = 50

# ==========================================
# 2. 数据加载与预处理
# ==========================================
# 图片文件名即为标签，例如 "2b827.png" -> 标签为 "2b827"
images = []
labels = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(DATA_DIR, filename)
        label = filename.split('.')[0]  # 去除后缀获取标签
        images.append(img_path)
        labels.append(label)

# 获取所有独立字符构建词汇表
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))
num_classes = len(characters)

# 字符到数字的映射 (留出 0 给 CTC Blank)
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# 划分训练集和测试集 (90% 训练, 10% 测试)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=TEST_SIZE, random_state=42)

def encode_single_sample(img_path, label):
    # 读取图片并转换为灰度图
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.transpose(img, perm=[1, 0, 2])  # 转置以匹配时间步长 (W, H, C)

    # 标签转数字
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    return {"image": img, "label": label}


# 构建 tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# ==========================================
# 3. 构建 CNN + CTC 模型
# ==========================================
class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # 计算训练时的 CTC Loss
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


def build_model():
    input_img = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # CNN 提取特征
    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 重塑特征图以适配 RNN 输入
    new_shape = ((IMG_WIDTH // 4), (IMG_HEIGHT // 4) * 64)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNN (双向 LSTM)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # 输出层 (类别数 + 1 个空白符)
    x = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2")(x)

    # CTC 损失层计算
    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam())
    return model


model = build_model()

# ==========================================
# 4. 训练回调
# ==========================================
class PrintLossCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {loss:.4f} - val_loss: {val_loss:.4f}")

# ==========================================
# 5. 训练模型
# ==========================================
print("开始训练模型...")
# 我们将 verbose 设为 0，关闭 keras 默认的冗长进度条，全权交由自定义 Callback 打印
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    verbose=0,
    callbacks=[PrintLossCallback()]
)

# ==========================================
# 6. 测试与评估指标计算 (Recall, Precision 等)
# ==========================================
# 提取用于预测的模型（去除了 CTC Loss 层）
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # 使用贪心搜索解码 CTC
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :5]

    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res.replace('[UNK]', '').strip())
    return output_text


print("\n开始测试并计算评估指标...")
true_labels = []
pred_labels = []

for batch in test_dataset:
    batch_images = batch["image"]
    batch_labels = batch["label"]

    # 预测
    preds = prediction_model.predict(batch_images, verbose=0)
    pred_texts = decode_batch_predictions(preds)

    # 获取真实标签
    for label in batch_labels:
        true_text = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        true_labels.append(true_text)

    pred_labels.extend(pred_texts)

# 统计指标
exact_matches = 0
y_true_chars = []
y_pred_chars = []

for true_text, pred_text in zip(true_labels, pred_labels):
    # 1. 序列级别的绝对匹配准确率
    if true_text == pred_text:
        exact_matches += 1

    # 2. 为计算字符级的 Precision/Recall，我们将字符串对齐填充（应对 CTC 输出长度变动）
    max_len = max(len(true_text), len(pred_text))
    t_chars = list(true_text.ljust(max_len, ' '))
    p_chars = list(pred_text.ljust(max_len, ' '))

    y_true_chars.extend(t_chars)
    y_pred_chars.extend(p_chars)

accuracy = exact_matches / len(true_labels)
# 计算宏平均(macro)的精确率、召回率和 F1 分数
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true_chars, y_pred_chars, average='macro', zero_division=0
)

print(f"--- 测试集评估结果 (共 {len(true_labels)} 张验证码) ---")
print(f"整体验证码完全匹配率 (Exact Match Accuracy): {accuracy * 100:.2f}%")
print(f"字符级精确率 (Precision): {precision:.4f}")
print(f"字符级召回率 (Recall): {recall:.4f}")
print(f"字符级 F1-Score: {f1:.4f}")