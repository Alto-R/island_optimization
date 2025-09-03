#!/bin/bash

CSV_FILE="chosen_island.csv"
MAIN_LOG="logs/main_log_cluster.log"

# --- 清理和初始化日志 ---
mkdir -p logs # 确保logs目录存在
> "$MAIN_LOG"  # 清空主日志文件

# --- 读取CSV文件并循环处理每个岛屿 ---
tail -n +2 "$CSV_FILE" | while IFS=',' read -r ID Long Lat Country Island Pop Geometry Region; do
    # 为日志文件定义一个清晰的前缀
    LOG_PREFIX="logs/log_${ID}"
    
    # 任务重试与执行函数
    # 参数: 1:TASK_NAME, 2:SCRIPT_NAME, 3:LOG_FILE
    run_task_with_retry() {
        local TASK_NAME=$1
        local SCRIPT_NAME=$2
        local LOG_FILE=$3
        local ATTEMPT=1
        local MAX_ATTEMPTS=2

        echo "Task '$TASK_NAME' started for Island ID: $ID at $(date +'%Y-%m-%d %H:%M:%S')" >> "$MAIN_LOG"
        
        while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
            echo "--- Attempt $ATTEMPT for task '$TASK_NAME' on Island $ID ---" >> "$LOG_FILE"
            # 执行Python脚本，传递参数，并将标准输出和错误都重定向到日志文件
            python3 "$SCRIPT_NAME" --island_lat "$Lat" --island_lon "$Long" --pop "$Pop" >> "$LOG_FILE" 2>&1
            
            # $? 变量会保存上一个命令的退出码。0代表成功。
            if [ $? -eq 0 ]; then
                echo "Task '$TASK_NAME' completed successfully for ID: $ID on attempt $ATTEMPT." >> "$MAIN_LOG"
                return 0 # 返回成功码 (0)
            else
                echo "Task '$TASK_NAME' failed for ID: $ID on attempt $ATTEMPT." >> "$MAIN_LOG"
                ATTEMPT=$((ATTEMPT + 1))
                if [ $ATTEMPT -le $MAX_ATTEMPTS ]; then
                    echo "Retrying in 10 seconds..." >> "$MAIN_LOG"
                    sleep 10
                fi
            fi
        done

        echo "Task '$TASK_NAME' failed permanently for ID: $ID after $MAX_ATTEMPTS attempts." >> "$MAIN_LOG"
        return 1 # 返回失败码 (1)
    }

    ISLAND_START_TIME=$(date +'%s') # 使用时间戳以便计算
    echo "======================================================================" >> "$MAIN_LOG"
    echo "Processing Island ID: $ID (Lat: $Lat, Lon: $Long) STARTED at $(date -d @$ISLAND_START_TIME +'%Y-%m-%d %H:%M:%S')" >> "$MAIN_LOG"
    echo "======================================================================" >> "$MAIN_LOG"

    # --- 串行执行聚类任务 ---
    run_task_with_retry "clustering" "cluster.py" "${LOG_PREFIX}_cluster.log"
    CLUSTER_EXIT_CODE=$? # 捕获任务的最终退出码

    # --- 任务执行完毕，开始统计和报告 ---
    ISLAND_END_TIME=$(date +'%s')
    ISLAND_DURATION=$((ISLAND_END_TIME - ISLAND_START_TIME))
    # 将秒转换为 时:分:秒 格式
    DURATION_FORMAT=$(printf "%02d:%02d:%02d" $((ISLAND_DURATION/3600)) $((ISLAND_DURATION%3600/60)) $((ISLAND_DURATION%60)))

    if [ $CLUSTER_EXIT_CODE -eq 0 ]; then
        echo "Island ID: $ID - ALL TASKS COMPLETED SUCCESSFULLY." >> "$MAIN_LOG"
    else
        echo "Island ID: $ID - A TASK FAILED. Check log: ${LOG_PREFIX}_cluster.log" >> "$MAIN_LOG"
    fi
    
    echo "Total processing time for Island $ID: $DURATION_FORMAT" >> "$MAIN_LOG"
    echo "" >> "$MAIN_LOG" # 添加空行以分隔不同岛屿的日志

done

echo "All islands processed. Script finished at $(date +'%Y-%m-%d %H:%M:%S')." >> "$MAIN_LOG"