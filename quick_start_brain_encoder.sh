#!/bin/bash
# Quick Start Script for Brain Encoder Training
# 快速启动脚本：用于Brain Encoder训练

set -e  # Exit on error

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示使用帮助
show_help() {
    cat << EOF
U-Mamba Brain Encoder 快速启动脚本

用法: $0 [选项]

选项:
    setup       - 设置环境和安装依赖
    prepare     - 准备数据集
    preprocess  - 预处理数据
    train       - 开始训练
    extract     - 提取特征
    all         - 执行所有步骤（setup -> prepare -> preprocess -> train）
    help        - 显示此帮助信息

示例:
    $0 setup              # 仅设置环境
    $0 train              # 仅训练模型
    $0 all                # 执行完整流程

配置:
    在运行前，请编辑脚本中的配置部分，设置数据路径等参数
EOF
}

# ============================================
# 配置部分 - 请根据你的环境修改
# ============================================

# 数据路径配置
SOURCE_IMAGE_DIR="/home/huawei/public/home/langdy/pipeline2/processed_3d_128_nii"  # 修改为你的原始图像目录
SOURCE_LABEL_DIR=""  # 如果有标签，设置此路径；否则留空

# 数据集配置
DATASET_ID=800
DATASET_NAME="BrainMRI"

# nnU-Net路径（通常不需要修改，会自动从环境变量或paths.py读取）
# NNUNET_RAW="/path/to/nnUNet_raw"
# NNUNET_PREPROCESSED="/path/to/nnUNet_preprocessed"
# NNUNET_RESULTS="/path/to/nnUNet_results"

# 训练配置
CONFIGURATION="3d_fullres"  # 2d, 3d_lowres, 3d_fullres, 3d_cascade
TRAINER="nnUNetTrainerBrainEncoder"
FOLD="all"  # all 或 0, 1, 2, 3, 4

# GPU配置
GPU_ID="4,5,6,7"  # 使用的GPU ID，多GPU用逗号分隔（如 "0,1,2,3"）
NUM_GPUS=4        # GPU数量（必须与GPU_ID中的数量一致）

# 特征提取配置
CHECKPOINT="checkpoint_final.pth"  # checkpoint_best.pth 或 checkpoint_final.pth
POOLING="--pooling"  # 留空则不池化
POOLING_METHOD="adaptive_avg"  # adaptive_avg, max, avg

# ============================================
# 函数定义
# ============================================

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."

    # 检查Python
    if ! command -v python &> /dev/null; then
        print_error "Python未安装"
        exit 1
    fi

    print_info "Python版本: $(python --version)"

    # 检查必要的Python包
    python -c "import torch" 2>/dev/null || {
        print_error "PyTorch未安装"
        exit 1
    }

    python -c "import SimpleITK" 2>/dev/null || {
        print_warning "SimpleITK未安装，将尝试安装..."
        pip install SimpleITK
    }

    print_info "依赖检查完成"
}

# 设置环境
setup_environment() {
    print_info "设置环境..."

    # 安装U-Mamba
    if [ ! -f "umamba/setup.py" ]; then
        print_error "请在U-Mamba根目录运行此脚本"
        exit 1
    fi

    print_info "安装U-Mamba..."
    cd umamba
    pip install -e . || {
        print_error "U-Mamba安装失败"
        exit 1
    }
    cd ..

    # 安装mamba-ssm
    print_info "检查mamba-ssm..."
    python -c "import mamba_ssm" 2>/dev/null || {
        print_info "安装mamba-ssm..."
        pip install mamba-ssm
    }

    # 安装其他依赖
    print_info "安装其他依赖..."
    pip install batchgenerators

    print_info "环境设置完成"
}

# 准备数据集
prepare_dataset() {
    print_info "准备数据集..."

    if [ ! -f "prepare_brain_dataset.py" ]; then
        print_error "prepare_brain_dataset.py 未找到"
        exit 1
    fi

    # 检查源数据目录
    if [ ! -d "$SOURCE_IMAGE_DIR" ]; then
        print_error "源图像目录不存在: $SOURCE_IMAGE_DIR"
        print_info "请编辑脚本中的 SOURCE_IMAGE_DIR 配置"
        exit 1
    fi

    print_info "源图像目录: $SOURCE_IMAGE_DIR"
    print_info "源标签目录: ${SOURCE_LABEL_DIR:-无}"
    print_info "数据集ID: $DATASET_ID"
    print_info "数据集名称: $DATASET_NAME"

    # 运行数据准备脚本
    print_info "运行数据准备脚本..."

    # 构建参数
    LABEL_ARG=""
    if [ -n "$SOURCE_LABEL_DIR" ]; then
        LABEL_ARG="--source_label_dir $SOURCE_LABEL_DIR"
    fi

    python prepare_brain_dataset.py \
        --source_image_dir "$SOURCE_IMAGE_DIR" \
        $LABEL_ARG \
        --dataset_id $DATASET_ID \
        --dataset_name "$DATASET_NAME"

    print_info "数据准备完成"
}

# 预处理数据
preprocess_data() {
    print_info "预处理数据..."

    print_info "运行 nnUNetv2_plan_and_preprocess..."

    nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity || {
        print_error "预处理失败"
        exit 1
    }

    print_info "预处理完成"
}

# 训练模型
train_model() {
    print_info "开始训练..."

    print_info "配置："
    print_info "  数据集ID: $DATASET_ID"
    print_info "  配置: $CONFIGURATION"
    print_info "  训练器: $TRAINER"
    print_info "  Fold: $FOLD"
    print_info "  GPU: $GPU_ID"
    print_info "  GPU数量: $NUM_GPUS"

    # 设置GPU
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    # 训练命令
    if [ $NUM_GPUS -gt 1 ]; then
        print_info "使用分布式训练（DDP），GPU数量: $NUM_GPUS"
        nnUNetv2_train $DATASET_ID $CONFIGURATION $FOLD \
            -tr $TRAINER \
            -num_gpus $NUM_GPUS || {
            print_error "训练失败"
            exit 1
        }
    else
        print_info "使用单GPU训练"
        nnUNetv2_train $DATASET_ID $CONFIGURATION $FOLD -tr $TRAINER || {
            print_error "训练失败"
            exit 1
        }
    fi

    print_info "训练完成"
}

# 提取特征
extract_features() {
    print_info "提取特征..."

    # 构建模型文件夹路径
    MODEL_FOLDER="nnUNet_results/Dataset${DATASET_ID}_${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIGURATION}/fold_${FOLD}"

    if [ ! -d "$MODEL_FOLDER" ]; then
        print_error "模型文件夹不存在: $MODEL_FOLDER"
        print_info "请先训练模型"
        exit 1
    fi

    print_info "模型文件夹: $MODEL_FOLDER"

    # 询问用户输入输出路径
    read -p "请输入要提取特征的图像文件夹路径: " IMAGE_FOLDER
    read -p "请输入特征输出文件夹路径: " OUTPUT_FOLDER

    if [ ! -d "$IMAGE_FOLDER" ]; then
        print_error "图像文件夹不存在: $IMAGE_FOLDER"
        exit 1
    fi

    mkdir -p "$OUTPUT_FOLDER"

    print_info "开始提取特征..."

    CUDA_VISIBLE_DEVICES=$GPU_ID python extract_brain_features.py \
        --model_folder "$MODEL_FOLDER" \
        --image_folder "$IMAGE_FOLDER" \
        --output_folder "$OUTPUT_FOLDER" \
        --checkpoint "$CHECKPOINT" \
        $POOLING \
        --pooling_method "$POOLING_METHOD" \
        --device cuda || {
        print_error "特征提取失败"
        exit 1
    }

    print_info "特征提取完成"
    print_info "输出目录: $OUTPUT_FOLDER"
}

# ============================================
# 主程序
# ============================================

# 检查参数
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 根据参数执行相应操作
case "$1" in
    setup)
        check_dependencies
        setup_environment
        print_info "✓ 环境设置完成"
        ;;

    prepare)
        prepare_dataset
        print_info "✓ 数据准备完成"
        ;;

    preprocess)
        preprocess_data
        print_info "✓ 数据预处理完成"
        ;;

    train)
        train_model
        print_info "✓ 模型训练完成"
        ;;

    extract)
        extract_features
        print_info "✓ 特征提取完成"
        ;;

    all)
        check_dependencies
        setup_environment
        prepare_dataset
        preprocess_data
        train_model
        print_info "✓ 所有步骤完成"
        print_info "下一步: 运行 '$0 extract' 提取特征"
        ;;

    help|--help|-h)
        show_help
        ;;

    *)
        print_error "未知选项: $1"
        show_help
        exit 1
        ;;
esac

print_info "完成！"
