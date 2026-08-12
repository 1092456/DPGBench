#!/bin/bash

# ================================================
# Comprehensive graph-attack test script
# Test all combinations of synthetic graph methods and attack modes.
# ================================================

# Configure colored terminal output.
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configure experiment parameters.
DATA_NAME=("p2p-Gnutella25")
NODE_TARGETS=3
EDGE_TARGETS=3
ATTACKS_PER_TARGET=20
EPSILON_VALUES="1,2,3,4,5,6,7,8,9,10,9999"

# Synthetic graph method list; all listed methods are recognized by run.py.
SYNTHETIC_METHODS=(
    "PrivGraph"
    "DGG"
    "Tmf"
    "PrivHRG"
    "PrivDPR"
    "SKG"
    "DP1K"
)

# Attack modes to evaluate.
ATTACK_MODES=("MIA" "AIA")

# Track aggregate test progress.
TOTAL_TESTS=$(( ${#SYNTHETIC_METHODS[@]} * ${#ATTACK_MODES[@]} ))
CURRENT_TEST=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# Create timestamped log files.
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="test_results_${TIMESTAMP}.log"
SUMMARY_FILE="test_summary_${TIMESTAMP}.csv"

# ================================================
# Function: print a colored message.
# ================================================
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# ================================================
# Function: write a message to the log file.
# ================================================
log_message() {
    local message=$1
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $message" >> "$LOG_FILE"
}

# ================================================
# Function: run one experiment configuration.
# ================================================
run_test() {
    local attack_mode=$1
    local synthetic_method=$2

    CURRENT_TEST=$((CURRENT_TEST + 1))

    print_message "$YELLOW" "\n=========================================="
    print_message "$YELLOW" "Test $CURRENT_TEST / $TOTAL_TESTS"
    print_message "$YELLOW" "Attack mode: $attack_mode"
    print_message "$YELLOW" "Synthesis method: $synthetic_method"
    print_message "$YELLOW" "Dataset: $DATA_NAME"
    print_message "$YELLOW" "==========================================\n"

    log_message "Starting test: $attack_mode - $synthetic_method"

    # Build the command for the selected attack and synthesis method.
    CMD="python run.py \
        --attack_mode $attack_mode \
        --data_name $DATA_NAME \
        --synthetic_method $synthetic_method \
        --node_targets $NODE_TARGETS \
        --edge_targets $EDGE_TARGETS \
        --attacks_per_target $ATTACKS_PER_TARGET \
        --epsilon_values $EPSILON_VALUES"

    print_message "$CYAN" "Executing command:"
    echo "$CMD"
    echo ""

    # Execute the configured run.py invocation.
    eval $CMD

    # Record the execution result in both the log and summary files.
    if [ $? -eq 0 ]; then
        print_message "$GREEN" "Test succeeded: $attack_mode - $synthetic_method"
        log_message "Test succeeded: $attack_mode - $synthetic_method"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "$attack_mode,$synthetic_method,success,$(date +"%Y-%m-%d %H:%M:%S")" >> "$SUMMARY_FILE"
    else
        print_message "$RED" "Test failed: $attack_mode - $synthetic_method"
        log_message "Test failed: $attack_mode - $synthetic_method"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "$attack_mode,$synthetic_method,failure,$(date +"%Y-%m-%d %H:%M:%S")" >> "$SUMMARY_FILE"
    fi

    echo ""
    sleep 2  # Wait briefly to reduce resource contention between runs.
}

# ================================================
# Main program starts.
# ================================================
clear
print_message "$PURPLE" "=========================================="
print_message "$PURPLE" "    Comprehensive graph-attack test script v1.0"
print_message "$PURPLE" "=========================================="
echo ""

print_message "$BLUE" "Test configuration:"
print_message "$BLUE" "------------------------------------------"
echo "Dataset: $DATA_NAME"
echo "Number of node targets: $NODE_TARGETS"
echo "Number of edge targets: $EDGE_TARGETS"
echo "Attacks per target: $ATTACKS_PER_TARGET"
echo "Privacy budgets: $EPSILON_VALUES"
echo "Synthetic graph methods: ${SYNTHETIC_METHODS[*]}"
echo "Attack modes: ${ATTACK_MODES[*]}"
echo "Total tests: $TOTAL_TESTS"
print_message "$BLUE" "------------------------------------------"
echo ""
print_message "$BLUE" "Log file: $LOG_FILE"
print_message "$BLUE" "Summary file: $SUMMARY_FILE"
echo ""

# Initialize the log file.
echo "Test log - $(date)" > "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Initialize the CSV summary file.
echo "attack_mode,synthesis_method,status,time" > "$SUMMARY_FILE"

# Ask for confirmation before launching the full batch.
read -p "Start tests? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_message "$RED" "Tests canceled"
    exit 1
fi

echo ""
print_message "$YELLOW" "Starting tests..."
log_message "Starting comprehensive tests"

# Record the start time for the full batch.
START_TIME=$(date +%s)

# Run all attack/method combinations.
for attack_mode in "${ATTACK_MODES[@]}"; do
    for synthetic_method in "${SYNTHETIC_METHODS[@]}"; do
        run_test "$attack_mode" "$synthetic_method"
    done
done

# Compute total elapsed time.
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

# ================================================
# Display the test-result summary.
# ================================================
clear
print_message "$PURPLE" "=========================================="
print_message "$PURPLE" "          Tests completed - result summary"
print_message "$PURPLE" "=========================================="
echo ""

print_message "$GREEN" "Success: $SUCCESS_COUNT / $TOTAL_TESTS"
print_message "$RED" "Failure: $FAIL_COUNT / $TOTAL_TESTS"
print_message "$CYAN" "Total elapsed time: ${DURATION_MIN}m ${DURATION_SEC}s"
echo ""

print_message "$BLUE" "Detailed results:"
print_message "$BLUE" "------------------------------------------"

# Read and display the summary file, skipping the CSV header.
if [ -f "$SUMMARY_FILE" ]; then
    tail -n +2 "$SUMMARY_FILE" | while IFS=',' read -r mode method status time; do
        if [ "$status" = "success" ]; then
            echo -e "${mode} - ${method}: ${GREEN}success${NC}"
        else
            echo -e "${mode} - ${method}: ${RED}failure${NC}"
        fi
    done
fi

print_message "$BLUE" "------------------------------------------"
echo ""
print_message "$GREEN" "Log file: $LOG_FILE"
print_message "$GREEN" "Summary file: $SUMMARY_FILE"
echo ""

# Save the final test configuration and aggregate result counts.
CONFIG_FILE="test_config_${TIMESTAMP}.txt"
cat > "$CONFIG_FILE" << EOF
Test configuration
================================
Date: $(date)
Dataset: $DATA_NAME
Number of node targets: $NODE_TARGETS
Number of edge targets: $EDGE_TARGETS
Attacks per target: $ATTACKS_PER_TARGET
Privacy budgets: $EPSILON_VALUES
Synthetic graph methods: ${SYNTHETIC_METHODS[*]}
Attack modes: ${ATTACK_MODES[*]}
Total tests: $TOTAL_TESTS
Success: $SUCCESS_COUNT
Failure: $FAIL_COUNT
Total elapsed time: ${DURATION_MIN}m ${DURATION_SEC}s
EOF

print_message "$CYAN" "Configuration file saved: $CONFIG_FILE"

# Warn the user when any configuration failed.
if [ $FAIL_COUNT -gt 0 ]; then
    print_message "$RED" "\nWarning: $FAIL_COUNT test(s) failed; please check the log file."
fi

print_message "$PURPLE" "\n=========================================="
