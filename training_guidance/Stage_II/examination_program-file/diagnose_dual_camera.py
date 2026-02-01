#!/usr/bin/env python3
"""
診斷雙相機問題
檢查系統是否正確識別兩個相機

執行: python diagnose_dual_camera.py
"""
import subprocess
import os

def run_cmd(cmd):
    """執行命令並返回輸出"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {e}"

def main():
    print("="*60)
    print("雙相機診斷工具")
    print("="*60)

    # 1. 檢查 /dev/video* 設備
    print("\n[1] 檢查 /dev/video* 設備")
    print("-"*40)
    output = run_cmd("ls -la /dev/video*")
    print(output)

    # 2. 檢查 /dev/media* 設備
    print("\n[2] 檢查 /dev/media* 設備")
    print("-"*40)
    output = run_cmd("ls -la /dev/media*")
    print(output)

    # 3. 檢查 v4l2 設備
    print("\n[3] 檢查 v4l2-ctl 設備列表")
    print("-"*40)
    output = run_cmd("v4l2-ctl --list-devices")
    print(output)

    # 4. 檢查 libcamera 設備
    print("\n[4] 檢查 libcamera 設備列表")
    print("-"*40)
    output = run_cmd("libcamera-hello --list-cameras")
    print(output)

    # 5. 檢查 I2C 設備
    print("\n[5] 檢查 I2C 設備 (bus 0)")
    print("-"*40)
    output = run_cmd("i2cdetect -y 0 2>/dev/null || echo 'i2cdetect not available or bus 0 not found'")
    print(output)

    print("\n[6] 檢查 I2C 設備 (bus 1)")
    print("-"*40)
    output = run_cmd("i2cdetect -y 1 2>/dev/null || echo 'i2cdetect not available or bus 1 not found'")
    print(output)

    print("\n[7] 檢查 I2C 設備 (bus 10)")
    print("-"*40)
    output = run_cmd("i2cdetect -y 10 2>/dev/null || echo 'Bus 10 not found'")
    print(output)

    print("\n[8] 檢查 I2C 設備 (bus 11)")
    print("-"*40)
    output = run_cmd("i2cdetect -y 11 2>/dev/null || echo 'Bus 11 not found'")
    print(output)

    # 6. 檢查 config.txt
    print("\n[9] 檢查 /boot/firmware/config.txt 相機設定")
    print("-"*40)
    try:
        with open("/boot/firmware/config.txt", "r") as f:
            lines = f.readlines()
            camera_lines = [l.strip() for l in lines if 'cam' in l.lower() or 'imx' in l.lower() or 'dtoverlay' in l.lower()]
            if camera_lines:
                for line in camera_lines:
                    print(line)
            else:
                print("未找到相機相關設定")
    except FileNotFoundError:
        # 嘗試舊路徑
        try:
            with open("/boot/config.txt", "r") as f:
                lines = f.readlines()
                camera_lines = [l.strip() for l in lines if 'cam' in l.lower() or 'imx' in l.lower() or 'dtoverlay' in l.lower()]
                if camera_lines:
                    for line in camera_lines:
                        print(line)
                else:
                    print("未找到相機相關設定")
        except FileNotFoundError:
            print("找不到 config.txt")

    # 7. 檢查 dmesg 相機相關訊息
    print("\n[10] 檢查 dmesg 相機相關訊息 (最近 20 條)")
    print("-"*40)
    output = run_cmd("dmesg | grep -i -E 'imx296|camera|cam|i2c' | tail -20")
    print(output if output.strip() else "無相關訊息")

    # 8. 嘗試用 Picamera2 列出相機
    print("\n[11] Picamera2 相機列表")
    print("-"*40)
    try:
        from picamera2 import Picamera2

        # 獲取全局相機列表
        cameras = Picamera2.global_camera_info()
        print(f"找到 {len(cameras)} 個相機:")
        for i, cam_info in enumerate(cameras):
            print(f"\n  Camera {i}:")
            for key, value in cam_info.items():
                print(f"    {key}: {value}")
    except Exception as e:
        print(f"Picamera2 錯誤: {e}")

    # 總結
    print("\n" + "="*60)
    print("診斷總結")
    print("="*60)
    print("""
常見問題與解決方案:

1. 只看到一個相機:
   - 檢查第二個相機的排線連接
   - 確認 /boot/firmware/config.txt 有正確的 dtoverlay 設定

2. Raspberry Pi 5 雙相機設定 (config.txt):
   # 相機 0 (CAM0 port)
   dtoverlay=imx296,cam0

   # 相機 1 (CAM1 port)
   dtoverlay=imx296,cam1

3. 如果兩個相機在同一個 I2C bus:
   - 需要修改其中一個的 I2C 地址
   - 或使用 I2C multiplexer

4. 重新啟動後檢查:
   sudo reboot
""")

if __name__ == "__main__":
    main()
