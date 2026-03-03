# Deep Learning Framework

基于 PyTorch 的深度学习框架模板

## 结构

- **src/core/** - 核心包 (可独立安装)
- **src/new_prj/** - 项目模板

## 快速开始

### 安装核心包

```bash
pip install -e src/core
```

### 使用模板创建新项目

```bash
# 1. 克隆项目模板
git clone <your-new-project-url>
cd your-new-project

# 2. 添加 DLF 为 submodule
git submodule add <DLF仓库URL> packages/dlf

# 3. 复制模板文件到根目录
cp -r packages/dlf/src/new_prj/* .
cp -r packages/dlf/src/new_prj/.* . 2>/dev/null || true

# 4. 安装依赖
pip install -e .

# 5. 开始训练
python -m dl.monkey_module
```

## 自定义

- 修改 Dockerfile & 镜像名
- 修改 .devcontainer 配置
- 基于 src/core 开发新模型