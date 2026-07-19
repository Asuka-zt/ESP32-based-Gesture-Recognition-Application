# ESP32-CAM OV3660 手势鼠标

## 1. 项目简介

本项目使用 AI-Thinker ESP32-CAM 和 OV3660 摄像头采集手部图像，通过局域网传输到 macOS。电脑端使用 MediaPipe 提取手部关键点，并使用 MobileNetV3-Small 迁移学习模型识别五种手势，实现鼠标移动、左击、拖拽、暂停和急停。

阶段状态：

- [x] 工程与 Python 3.12 环境
- [x] OV3660 固件与 MJPEG 接口（需接板实测）
- [x] 视频接收、断流重连和网页监控
- [x] 数据采集与训练工具（需真实数据训练）
- [x] 实时识别与模型安全校验
- [x] macOS 鼠标控制（需授予权限实测）
- [x] 自动化测试、启动脚本和项目说明书

## 2. 功能展示

- ESP32-CAM 提供 OV3660 实时 MJPEG 视频流和设备状态。
- 本地网页显示画面、FPS、延迟、RSSI、识别类别和控制状态。
- 网页创建人员/场次并采集五类手势数据。
- MobileNetV3-Small 训练输出权重、元数据、曲线和混淆矩阵。
- 指向移动光标，OK/捏合完成单击或拖拽，张掌启停，握拳急停。
- 断流、丢手、异常退出和急停均强制释放鼠标左键。

## 3. 系统架构

```text
OV3660
  ↓ JPEG
ESP32-CAM ── /status
  ↓ /stream MJPEG
最新帧缓冲（旧帧覆盖、断流重连）
  ↓
MediaPipe 单手关键点 ──→ 手部裁剪 ──→ MobileNetV3-Small
  ↓                                      ↓
食指/捏合几何信息                  五类概率与稳定类别
  └──────────────────┬───────────────────┘
                     ↓
             安全鼠标控制状态机
                     ↓
                macOS Quartz
```

## 4. 硬件清单与连接

- AI-Thinker ESP32-CAM 开发板
- OV3660 摄像头模组
- ESP32-CAM-MB USB 下载底板
- 稳定 5V/1A 或更高规格电源
- 可靠 USB 数据线

固件采用 AI-Thinker 引脚定义。启动时读取摄像头 PID，只有确认 OV3660 后才启动 HTTP 服务。摄像头排线必须断电安装，金属触点方向以开发板插座标识为准。

## 5. OV3660 固件烧录

```bash
cp firmware/include/wifi_secrets.example.h firmware/include/wifi_secrets.h
```

修改 `wifi_secrets.h` 中的 Wi-Fi 名称和密码，然后执行：

```bash
uv tool run platformio run --project-dir firmware
uv tool run platformio run --project-dir firmware --target upload
uv tool run platformio device monitor --baud 115200
```

串口应显示 `Detected camera sensor: OV3660`。设备接口：

```text
http://esp32cam.local/status
http://esp32cam.local/stream
```

有 PSRAM 时默认 VGA、双帧缓冲、JPEG 质量 12；无 PSRAM 时降级为 QVGA 单帧缓冲。固件支持 Wi-Fi 自动重连，并在连续摄像头采集失败后重启恢复。

## 6. 软件环境安装

需要 macOS、`uv` 和可用的局域网。

```bash
uv python install 3.12
uv sync --all-extras
uv run pytest
uv run ruff check .
```

完整验证：

```bash
./scripts/verify.sh
```

## 7. 配置说明

```bash
cp .env.example .env
```

主要配置：

| 变量 | 用途 |
|---|---|
| `GESTURE_MOUSE_ESP32_BASE_URL` | ESP32 地址，mDNS 不可用时填写设备 IP |
| `GESTURE_MOUSE_MODEL_PATH` | PyTorch 权重路径 |
| `GESTURE_MOUSE_MODEL_METADATA_PATH` | 模型元数据路径 |
| `GESTURE_MOUSE_DATASET_ROOT` | 数据集根目录 |
| `GESTURE_MOUSE_ENABLE_MOUSE_CONTROL` | 是否允许创建 macOS 鼠标控制器，默认 `false` |

`.env`、`wifi_secrets.h`、数据集、模型和训练产物不会提交 Git。

## 8. 数据采集

五类标准手势：

| 类别 | 标准动作 |
|---|---|
| `point` | 仅伸出食指 |
| `ok` | 拇指与食指捏合 |
| `palm` | 五指自然展开 |
| `fist` | 完全握拳 |
| `v` | 食指与中指形成 V |

启动服务后访问 `http://127.0.0.1:8000`，使用网页的数据采集区创建场次并保存当前手部。数据目录为：

```text
data/<gesture>/<person>/<session>/*.jpg
```

至少采集 3 名操作者、左右手、2 种背景、多个距离和不同光照。推荐每人每类 150–200 张，总量约 2250–3000 张。训练、验证和测试按 `person/session` 分组，禁止同一连拍场次跨集合。

## 9. 模型训练

```bash
uv run gesture-train --data data --output models --epochs 20 --batch-size 32
```

自动选择 Apple MPS 或 CPU。输出：

- `gesture_mobilenet_v3.pt`
- `gesture_mobilenet_v3.json`
- `training_curves.png`
- `confusion_matrix.png`

正式控制前，独立测试集准确率目标为 90% 或以上。

## 10. 实时识别

```bash
./scripts/run.sh
```

网页地址为 `http://127.0.0.1:8000`，预测接口为 `GET /api/prediction`。模型元数据必须包含正确的五类顺序、输入尺寸、归一化参数和置信度阈值。模型缺失或不匹配时仍可查看视频，但不能启用鼠标控制。

稳定类别要求最近 5 帧至少获得 4 票；低置信度、无手和检测失败均输出未知状态。

## 11. 鼠标控制

在 `.env` 设置：

```text
GESTURE_MOUSE_ENABLE_MOUSE_CONTROL=true
```

在“系统设置 → 隐私与安全性 → 辅助功能”中允许启动服务的终端或 Python 进程控制电脑，然后重启服务。

| 手势 | 动作 |
|---|---|
| 指向 | 食指尖绝对映射到主屏幕并移动光标 |
| OK/捏合 | 捏合按下左键，松开单击；保持并移动为拖拽 |
| 张掌约 1 秒 | 切换启用和暂停 |
| 握拳 | 立即暂停并释放左键 |
| V | 仅展示识别结果 |

系统启动后默认暂停。网页急停按钮或网页获得焦点时按 Esc 会立即停止控制。丢手、断流、退出和异常均执行安全释放。

## 12. API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 服务健康检查 |
| GET | `/api/status` | ESP32、视频、识别和控制状态 |
| GET | `/api/live` | 带识别标注的 MJPEG 视频 |
| GET | `/api/prediction` | 最新手势预测 |
| POST | `/api/dataset/sessions` | 创建数据采集场次 |
| POST | `/api/dataset/capture` | 保存当前手部裁剪图 |
| POST | `/api/control/enable` | 启用鼠标控制 |
| POST | `/api/control/disable` | 暂停鼠标控制 |
| POST | `/api/control/emergency-stop` | 急停并释放左键 |

ESP32 提供 `GET /stream` 和 `GET /status`。

## 13. 测试与性能指标

```bash
uv run pytest
uv run ruff check .
uv tool run platformio run --project-dir firmware
```

目标指标：

- 多人多背景测试准确率不低于 90%。
- 正常局域网端到端延迟不超过 500 ms。
- 连续运行 30 分钟无左键卡死。
- Wi-Fi 恢复后无需重启电脑端服务。

自动化测试覆盖最新帧覆盖、路径安全、数据分组隔离、预测稳定和鼠标按下/释放平衡。摄像头画质、Wi-Fi 恢复、准确率、延迟和长时间运行必须在真实硬件及真实数据上人工验收。

## 14. 常见问题

- **检测到的不是 OV3660**：检查摄像头型号、排线方向和模组兼容性；固件会拒绝启动服务。
- **频繁重启或花屏**：优先更换稳定 5V 电源和 USB 线，并降低分辨率。
- **网页显示 offline**：确认 Mac 与 ESP32 在同一局域网，必要时使用设备 IP 替代 mDNS。
- **识别结果跳变**：增加对应人员和背景数据，检查运动模糊，并查看混淆矩阵。
- **控制接口返回 403**：授予 macOS 辅助功能权限并重启终端与服务。
- **控制接口返回 409**：模型缺失、损坏或类别元数据不匹配。

## 15. 安全停止与故障恢复

- 启动后默认暂停。
- 握拳、网页急停或网页内 Esc 立即暂停。
- 视频断流或持续丢手自动释放左键并暂停。
- 鼠标后端异常时进入暂停状态。
- 服务退出时通过 `finally` 路径释放左键。
- 如果仍出现异常，可关闭服务进程；macOS 不会保留进程已经退出后的软件鼠标事件。

## 16. 项目目录结构

```text
firmware/           ESP32-CAM PlatformIO 固件
src/gesture_mouse/  API、视频、识别、训练和鼠标控制
tests/              自动化测试
scripts/            启动和完整验证脚本
data/               本地数据集，不提交 Git
models/             模型和指标，不提交 Git
artifacts/          其他运行产物，不提交 Git
```

每个阶段必须同时更新实现、测试和 README，验收通过后创建独立本地 Git commit。

## 实施提交记录

- 阶段 1：OV3660 固件完成静态编译验收；真实传感器 PID、推流和断网恢复待接板验证。
- 阶段 2：完成最新帧缓冲、MJPEG 接收、断流重连、状态 API 和本地网页仪表盘。
- 阶段 3：完成安全数据目录、网页采集接口、分组切分和 MobileNetV3 训练产物。
