# Deep Learning Framework

基于 PyTorch 的深度学习框架模板

## 结构

- **src/dlf/** - 核心包 (可独立安装)
- **src/new_prj/** - 项目模板

## 快速开始

### 安装核心包

```bash
pip install -e packages/dlf
```

### 使用模板创建新项目

```bash
# 1. 进入新项目
cd your-new-project

# 2. 添加 DLF 为 submodule
git submodule add https://github.com/Penrose0v0/DeepLearningFramework.git packages/dlf
git submodule update --init --recursive

# 3. 复制模板文件到根目录
cp -r packages/dlf/src/new_prj/. .

# 4. 安装dlf核心包
pip install -e packages/dlf

# 5. 安装项目包
pip install -e .

```

## 自定义

- 修改 Dockerfile & 镜像名
- 修改 .devcontainer 配置
- 基于 src/core 开发新模型