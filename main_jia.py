# from OnnxDetect import DisasterDetector
# import cv2
# import time
# import RPi.GPIO as GPIO
# from xgoedu import XGOEDU

# def cleanup_gpio():
#     try:
#         GPIO.cleanup()
#     except:
#         pass

# def main_function():
#     # 确保在初始化前清理GPIO
#     cleanup_gpio()
    
#     try:
#         # 初始化XGOEDU
#         XGO_edu = XGOEDU()
        
#         # 1. 初始化检测器 (只需一次)
#         detector = DisasterDetector('/home/pi/四足机器人案例123/动物识别/best.onnx')

#         # 2. 主循环
#         while True:
#             # 你的其他主函数代码...
#             print("正在执行主任务...")
#             time.sleep(1)
            
#             # 3. 当需要检测时:
#             # 打开摄像头
#             XGO_edu.xgoCamera(True)
#             # 拍摄照片并保存为默认名称
#             XGO_edu.xgoTakePhoto(filename="captured_image")
#             img_path = "/home/pi/xgoPictures/captured_image.jpg"
#             time.sleep(2)  # 等待摄像头处理时间

#             try:
#                 img = cv2.imread(img_path)
#                 if img is None:
#                     print("无法读取图像，请检查路径")
#                     continue
                    
#                 # 调用检测
#                 result = detector.detect(img)
                
#                 if result:
#                     disaster, confidence = result
#                     print(f"检测到灾害: {disaster} (置信度: {confidence:.2f})")
#                     # 根据灾害类型采取行动
#                     if disaster == 'Fire':
#                         trigger_Fire_protocol(XGO_edu)
#                     elif disaster == 'Flood':
#                         trigger_Flood_protocol(XGO_edu)
#                     elif disaster == 'maoding':
#                         trigger_maoding_protocol(XGO_edu)
#                     elif disaster == 'Landslide':
#                         trigger_Landslide_protocol(XGO_edu)
#                     elif disaster == 'Explosion':
#                         trigger_Explosion_protocol(XGO_edu)
#                 else:
#                     print("未检测到灾害")
                    
#             except Exception as e:
#                 print(f"处理出错: {str(e)}")
                
#     except KeyboardInterrupt:
#         print("程序被用户中断")
#     except Exception as e:
#         print(f"程序运行出错: {str(e)}")
#     finally:
#         # 确保程序退出时清理资源
#         cleanup_gpio()

# def trigger_Fire_protocol(xgo):
#     print("此区域发生火灾")
#     xgo.xgoSpeaker('Fire.wav')
    
# def trigger_Flood_protocol(xgo):
#     print("此区域发生水灾")
#     xgo.xgoSpeaker('Flood.wav')

# def trigger_maoding_protocol(xgo):
#     print("此区域发生冒顶")
#     xgo.xgoSpeaker('Caving.wav')

# def trigger_Landslide_protocol(xgo):
#     print("此区域发生塌方")
#     xgo.xgoSpeaker('Landslide.wav')

# def trigger_Explosion_protocol(xgo):
#     print("此区域发生爆炸")
#     xgo.xgoSpeaker('Explosion.wav')
    
# if __name__ == '__main__':
#     main_function()
from OnnxDetect import DisasterDetector
import cv2
import time
import RPi.GPIO as GPIO
from xgoedu import XGOEDU
from xgolib import XGO
import os
import sys
import subprocess
dog = XGO("xgomini")
def hardware_full_reset():
    """彻底硬件复位"""
    try:
        # 1. 清理GPIO
        GPIO.cleanup()
        
        # 2. 杀死可能冲突的进程
        subprocess.run(['sudo', 'killall', '-9', 'python3'], stderr=subprocess.DEVNULL)
        subprocess.run(['sudo', 'killall', '-9', 'libcamera'], stderr=subprocess.DEVNULL)
        
        # 3. 重置摄像头
        subprocess.run(['sudo', 'systemctl', 'restart', 'camera-streamer'], stderr=subprocess.DEVNULL)
        
        # 4. 等待硬件稳定
        time.sleep(2.5)
    except Exception as e:
        print(f"硬件复位出错: {str(e)}")

def safe_camera_capture(xgo, retries=3):
    """安全拍照函数"""
    for attempt in range(retries):
        try:
            # 确保摄像头开启
            xgo.xgoCamera(True)
            time.sleep(1)  # 给摄像头足够启动时间
            
            # 拍照
            xgo.xgoTakePhoto(filename="captured_image")
            time.sleep(1)  # 确保文件写入完成
            
            img_path = "/home/pi/xgoPictures/captured_image.jpg"
            img = cv2.imread(img_path)
            
            if img is not None and img.size != 0:
                return img
                
        except Exception as e:
            print(f"拍照尝试 {attempt + 1} 失败: {str(e)}")
            if attempt == retries - 1:
                raise
            time.sleep(2)
    
    raise RuntimeError("无法获取有效图像")

def main_function():
    # 彻底硬件复位
    hardware_full_reset()
    
    try:
        # 初始化XGOEDU（最多尝试2次）
        for attempt in range(2):
            try:
                XGO_edu = XGOEDU()
                break
            except Exception as e:
                print(f"XGO初始化尝试 {attempt + 1} 失败: {str(e)}")
                if attempt == 1:
                    print("XGO初始化彻底失败，请检查硬件")
                    sys.exit(1)
                hardware_full_reset()
        
        # 初始化检测器
        detector = DisasterDetector('/home/pi/四足机器人案例123/动物识别/best.onnx')
        
        # 主循环
        while True:
            print("正在执行主任务...")
            
            try:
                # 安全拍照
                img = safe_camera_capture(XGO_edu)
                
                # 灾害检测
                result = detector.detect(img)
                
                # 处理结果
                if result:
                    disaster, confidence = result
                    print(f"检测到灾害: {disaster} (置信度: {confidence:.2f})")
                    #模型错误，爆炸和塌方互换，冒顶和火灾互换
                    protocols = {
                        'Fire': ('Fire.wav', "火灾"),
                        'Flood': ('Flood.wav', "水灾"),
                        'Caving': ('Caving.wav', "冒顶"),
                        'Landslide': ('Landslide.wav', "塌方"),
                        'Explosion': ('Explosion.wav', "爆炸")
                    }
                    
                    if disaster in protocols:
                        sound, name = protocols[disaster]
                        print(f"此区域发生{name}")
                        try:
                            XGO_edu.xgoSpeaker(sound)
                        except Exception as e:
                            print(f"播放音频失败: {str(e)}")
                    # 检测到灾害后跳出循环并结束函数
                    break
                else:
                    print("未检测到灾害")
                    # 未检测到灾害时执行移动指令
                    try:
                        dog.move_x(-2)  # 移动指令，参数可根据需要调整##改
                        print("执行移动指令")
                    except Exception as e:
                        print(f"移动失败: {str(e)}")
                    
                time.sleep(1)  # 正常循环间隔
                
            except Exception as e:
                print(f"主循环出错: {str(e)}")
                time.sleep(5)  # 出错后延长等待
                hardware_full_reset()  # 出错后复位硬件
                
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序致命错误: {str(e)}")
    finally:
        hardware_full_reset()
        print("程序结束")

if __name__ == '__main__':
    main_function()