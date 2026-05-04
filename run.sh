#!/bin/bash

# ================================================
# 图攻击全面测试脚本
# 测试所有合成图方法和攻击模式的组合
# ================================================

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 配置参数
DATA_NAME=("p2p-Gnutella25")
NODE_TARGETS=3
EDGE_TARGETS=3
ATTACKS_PER_TARGET=20
EPSILON_VALUES="1,2,3,4,5,6,7,8,9,10,9999"

# 合成图方法列表（现在所有方法都会被识别）
SYNTHETIC_METHODS=(
    "PrivGraph"
    "DGG"
    "Tmf"
    "PrivHRG"
    "PrivDPR"
    "SKG"
    "DP1K"
)

# 攻击模式列表
ATTACK_MODES=("MIA" "AIA")

# 计数器
TOTAL_TESTS=$(( ${#SYNTHETIC_METHODS[@]} * ${#ATTACK_MODES[@]} ))
CURRENT_TEST=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# 日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="test_results_${TIMESTAMP}.log"
SUMMARY_FILE="test_summary_${TIMESTAMP}.csv"

# ================================================
# 函数：打印带颜色的消息
# ================================================
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# ================================================
# 函数：记录日志
# ================================================
log_message() {
    local message=$1
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $message" >> "$LOG_FILE"
}

# ================================================
# 函数：运行单个测试
# ================================================
run_test() {
    local attack_mode=$1
    local synthetic_method=$2

    CURRENT_TEST=$((CURRENT_TEST + 1))

    print_message "$YELLOW" "\n=========================================="
    print_message "$YELLOW" "测试 $CURRENT_TEST / $TOTAL_TESTS"
    print_message "$YELLOW" "攻击模式: $attack_mode"
    print_message "$YELLOW" "合成方法: $synthetic_method"
    print_message "$YELLOW" "数据集: $DATA_NAME"
    print_message "$YELLOW" "==========================================\n"

    log_message "开始测试: $attack_mode - $synthetic_method"

    # 构建命令
    CMD="python run.py \
        --attack_mode $attack_mode \
        --data_name $DATA_NAME \
        --synthetic_method $synthetic_method \
        --node_targets $NODE_TARGETS \
        --edge_targets $EDGE_TARGETS \
        --attacks_per_target $ATTACKS_PER_TARGET \
        --epsilon_values $EPSILON_VALUES"

    print_message "$CYAN" "执行命令:"
    echo "$CMD"
    echo ""

    # 执行命令
    eval $CMD

    # 检查执行结果
    if [ $? -eq 0 ]; then
        print_message "$GREEN" "✓ 测试成功: $attack_mode - $synthetic_method"
        log_message "测试成功: $attack_mode - $synthetic_method"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # 记录到摘要文件
        echo "$attack_mode,$synthetic_method,成功,$(date +"%Y-%m-%d %H:%M:%S")" >> "$SUMMARY_FILE"
    else
        print_message "$RED" "✗ 测试失败: $attack_mode - $synthetic_method"
        log_message "测试失败: $attack_mode - $synthetic_method"
        FAIL_COUNT=$((FAIL_COUNT + 1))

        # 记录到摘要文件
        echo "$attack_mode,$synthetic_method,失败,$(date +"%Y-%m-%d %H:%M:%S")" >> "$SUMMARY_FILE"
    fi

    echo ""
    sleep 2  # 等待2秒，避免资源冲突
}

# ================================================
# 主程序开始
# ================================================
clear
print_message "$PURPLE" "=========================================="
print_message "$PURPLE" "    图攻击全面测试脚本 v1.0"
print_message "$PURPLE" "=========================================="
echo ""

print_message "$BLUE" "测试配置:"
print_message "$BLUE" "------------------------------------------"
echo "数据集: $DATA_NAME"
echo "节点目标数: $NODE_TARGETS"
echo "边目标数: $EDGE_TARGETS"
echo "攻击次数/目标: $ATTACKS_PER_TARGET"
echo "隐私预算: $EPSILON_VALUES"
echo "合成图方法: ${SYNTHETIC_METHODS[*]}"
echo "攻击模式: ${ATTACK_MODES[*]}"
echo "总测试数: $TOTAL_TESTS"
print_message "$BLUE" "------------------------------------------"
echo ""
print_message "$BLUE" "日志文件: $LOG_FILE"
print_message "$BLUE" "摘要文件: $SUMMARY_FILE"
echo ""

# 初始化日志文件
echo "测试日志 - $(date)" > "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 初始化摘要文件
echo "攻击模式,合成方法,状态,时间" > "$SUMMARY_FILE"

# 询问用户是否继续
read -p "是否开始测试? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_message "$RED" "测试已取消"
    exit 1
fi

echo ""
print_message "$YELLOW" "开始测试..."
log_message "开始全面测试"

# 记录开始时间
START_TIME=$(date +%s)

# 循环运行所有测试
for attack_mode in "${ATTACK_MODES[@]}"; do
    for synthetic_method in "${SYNTHETIC_METHODS[@]}"; do
        run_test "$attack_mode" "$synthetic_method"
    done
done

# 计算总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

# ================================================
# 显示测试结果摘要
# ================================================
clear
print_message "$PURPLE" "=========================================="
print_message "$PURPLE" "          测试完成 - 结果摘要"
print_message "$PURPLE" "=========================================="
echo ""

print_message "$GREEN" "成功: $SUCCESS_COUNT / $TOTAL_TESTS"
print_message "$RED" "失败: $FAIL_COUNT / $TOTAL_TESTS"
print_message "$CYAN" "总耗时: ${DURATION_MIN}分${DURATION_SEC}秒"
echo ""

print_message "$BLUE" "详细结果:"
print_message "$BLUE" "------------------------------------------"

# 读取并显示摘要文件内容
if [ -f "$SUMMARY_FILE" ]; then
    # 跳过标题行
    tail -n +2 "$SUMMARY_FILE" | while IFS=',' read -r mode method status time; do
        if [ "$status" = "成功" ]; then
            echo -e "${mode} - ${method}: ${GREEN}✓ 成功${NC}"
        else
            echo -e "${mode} - ${method}: ${RED}✗ 失败${NC}"
        fi
    done
fi

print_message "$BLUE" "------------------------------------------"
echo ""
print_message "$GREEN" "日志文件: $LOG_FILE"
print_message "$GREEN" "摘要文件: $SUMMARY_FILE"
echo ""

# 保存测试配置
CONFIG_FILE="test_config_${TIMESTAMP}.txt"
cat > "$CONFIG_FILE" << EOF
测试配置信息
================================
日期: $(date)
数据集: $DATA_NAME
节点目标数: $NODE_TARGETS
边目标数: $EDGE_TARGETS
攻击次数/目标: $ATTACKS_PER_TARGET
隐私预算: $EPSILON_VALUES
合成图方法: ${SYNTHETIC_METHODS[*]}
攻击模式: ${ATTACK_MODES[*]}
总测试数: $TOTAL_TESTS
成功: $SUCCESS_COUNT
失败: $FAIL_COUNT
总耗时: ${DURATION_MIN}分${DURATION_SEC}秒
EOF

print_message "$CYAN" "配置文件已保存: $CONFIG_FILE"

# 如果有失败的测试，显示警告
if [ $FAIL_COUNT -gt 0 ]; then
    print_message "$RED" "\n⚠ 警告: 有 $FAIL_COUNT 个测试失败，请检查日志文件"
fi

print_message "$PURPLE" "\n=========================================="