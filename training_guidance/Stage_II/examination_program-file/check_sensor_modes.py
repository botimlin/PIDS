#!/usr/bin/env python3
"""
查看 IMX296 支援的所有 sensor 模式
用於確認是否支援 raw 輸出

執行: python check_sensor_modes.py
"""
from picamera2 import Picamera2

def check_camera(cam_id):
    print(f"\n{'='*60}")
    print(f"Camera {cam_id}")
    print('='*60)

    try:
        cam = Picamera2(cam_id)

        print("\n[Sensor Modes]")
        for i, mode in enumerate(cam.sensor_modes):
            print(f"\nMode {i}:")
            for key, value in mode.items():
                print(f"  {key}: {value}")

        print("\n[Camera Properties]")
        props = cam.camera_properties
        for key, value in props.items():
            print(f"  {key}: {value}")

        # 測試 raw 配置
        print("\n[Testing RAW Configuration]")
        try:
            config = cam.create_still_configuration(raw={})
            print(f"  RAW config supported!")
            print(f"  Raw stream: {config.get('raw', 'N/A')}")
        except Exception as e:
            print(f"  RAW config error: {e}")

        cam.close()

    except Exception as e:
        print(f"Error opening camera {cam_id}: {e}")

if __name__ == '__main__':
    print("IMX296 Sensor Mode Checker")
    print("="*60)

    # 檢查兩個相機
    check_camera(0)
    check_camera(1)

    print("\n" + "="*60)
    print("Done!")
