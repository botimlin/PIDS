@echo off
REM PIDS Training Script - Stage I
REM Usage: run_training.bat

echo ========================================
echo PIDS Stage I Training
echo ========================================

REM Set data and output paths
set DATA_DIR=..\rendering_v3\output\output
set OUTPUT_DIR=.\checkpoints

REM Check if data directory exists
if not exist "%DATA_DIR%" (
    echo Error: Data directory not found: %DATA_DIR%
    pause
    exit /b 1
)

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Data directory: %DATA_DIR%
echo Output directory: %OUTPUT_DIR%
echo.

REM Run training
python train_pids.py ^
    --data_dir "%DATA_DIR%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --batch_size 4 ^
    --num_steps 50000 ^
    --lr 0.0002 ^
    --iters 12 ^
    --crop_height 320 ^
    --crop_width 480 ^
    --glass_weight 2.0 ^
    --mixed_precision ^
    --print_freq 100 ^
    --val_freq 1 ^
    --save_freq 5

echo.
echo Training completed!
pause
