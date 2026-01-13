#!/bin/bash
# 训练监控脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DATASET_ID=800
DATASET_NAME="BrainMRI"
TRAINER="nnUNetTrainerBrainEncoder"
CONFIG="3d_fullres"

# 结果目录
RESULT_DIR="$HOME/U-Mamba/data/nnUNet_results/Dataset${DATASET_ID}_${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIG}/fold_all"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Brain Encoder 训练监控${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查训练是否在运行
if pgrep -f "nnUNetv2_train.*${DATASET_ID}" > /dev/null; then
    echo -e "${GREEN}✓ 训练进程正在运行${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠ 未检测到训练进程${NC}"
    echo ""
fi

# 检查结果目录是否存在
if [ ! -d "$RESULT_DIR" ]; then
    echo -e "${YELLOW}训练尚未开始或结果目录不存在${NC}"
    echo "预期目录: $RESULT_DIR"
    exit 1
fi

echo -e "${BLUE}结果目录:${NC} $RESULT_DIR"
echo ""

# 显示checkpoint信息
echo -e "${BLUE}=== Checkpoint 状态 ===${NC}"
if [ -f "$RESULT_DIR/checkpoint_latest.pth" ]; then
    echo -e "${GREEN}✓ checkpoint_latest.pth${NC} ($(ls -lh "$RESULT_DIR/checkpoint_latest.pth" | awk '{print $5}'))"
fi
if [ -f "$RESULT_DIR/checkpoint_best.pth" ]; then
    echo -e "${GREEN}✓ checkpoint_best.pth${NC} ($(ls -lh "$RESULT_DIR/checkpoint_best.pth" | awk '{print $5}'))"
fi
if [ -f "$RESULT_DIR/checkpoint_final.pth" ]; then
    echo -e "${GREEN}✓ checkpoint_final.pth${NC} ($(ls -lh "$RESULT_DIR/checkpoint_final.pth" | awk '{print $5}'))"
    echo -e "${GREEN}训练已完成！${NC}"
fi
echo ""

# 显示最新的训练日志
echo -e "${BLUE}=== 最新训练进度 ===${NC}"
LOG_FILE=$(ls -t "$RESULT_DIR"/training_log*.txt 2>/dev/null | head -1)
if [ -f "$LOG_FILE" ]; then
    echo "日志文件: $LOG_FILE"
    echo ""

    # 提取最新的epoch信息
    tail -n 50 "$LOG_FILE" | grep -E "Epoch|train_loss|val_loss|Pseudo dice|took" | tail -20
    echo ""

    # 显示当前epoch
    CURRENT_EPOCH=$(tail -n 100 "$LOG_FILE" | grep -oP "Epoch \K\d+" | tail -1)
    if [ -n "$CURRENT_EPOCH" ]; then
        echo -e "${GREEN}当前 Epoch: $CURRENT_EPOCH / 1000${NC}"
        PROGRESS=$((CURRENT_EPOCH * 100 / 1000))
        echo -e "${GREEN}进度: $PROGRESS%${NC}"
    fi
else
    echo -e "${YELLOW}未找到训练日志${NC}"
fi
echo ""

# GPU使用情况
echo -e "${BLUE}=== GPU 使用情况 ===${NC}"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | \
    awk -F, '{printf "GPU %s (%s): %s | 显存: %s /%s\n", $1, $2, $3, $4, $5}'
echo ""

# 选项菜单
echo -e "${BLUE}=== 操作选项 ===${NC}"
echo "1. 实时查看训练日志"
echo "2. 显示训练曲线图路径"
echo "3. 显示所有checkpoint"
echo "4. 监控GPU使用"
echo "5. 退出"
echo ""

read -p "请选择操作 (1-5): " choice

case $choice in
    1)
        echo "实时查看日志 (Ctrl+C 退出)..."
        tail -f "$LOG_FILE"
        ;;
    2)
        echo "训练曲线图: $RESULT_DIR/progress.png"
        if [ -f "$RESULT_DIR/progress.png" ]; then
            echo "文件大小: $(ls -lh "$RESULT_DIR/progress.png" | awk '{print $5}')"
            echo "最后修改: $(ls -l "$RESULT_DIR/progress.png" | awk '{print $6, $7, $8}')"
        else
            echo "文件尚未生成"
        fi
        ;;
    3)
        echo "所有checkpoint文件:"
        ls -lh "$RESULT_DIR"/*.pth 2>/dev/null || echo "未找到checkpoint文件"
        ;;
    4)
        echo "实时监控GPU (Ctrl+C 退出)..."
        watch -n 1 nvidia-smi
        ;;
    5)
        echo "退出"
        ;;
    *)
        echo "无效选择"
        ;;
esac
