#!/usr/bin/env python
# -*- coding: utf-8 -*-
import onnxruntime
import numpy as np
import cv2
from xgoedu import XGOEDU
XGO_edu = XGOEDU()

CLASSES = ['Fire','Landslide','Caving','Flood','Explosion']

class YOLOV5():
    def __init__(self, onnxpath):
        self.onnx_session = onnxruntime.InferenceSession(onnxpath)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    def get_input_name(self):
        return [node.name for node in self.onnx_session.get_inputs()]

    def get_output_name(self):
        return [node.name for node in self.onnx_session.get_outputs()]

    def get_input_feed(self, img_tensor):
        return {name: img_tensor for name in self.input_name}

    def inference(self, image):
        # Process single image to match batch size of 8
        or_img = cv2.resize(image, (640, 640))
        img = or_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        batch_images = np.stack([img]*8, axis=0)  # Shape [8, 3, 640, 640]
        
        input_feed = self.get_input_feed(batch_images)
        pred = self.onnx_session.run(None, input_feed)[0]
        return pred, or_img  # Return prediction and original resized image

class transTool():
    @staticmethod
    def xywh2xyxy(x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    @staticmethod
    def nms(dets, thresh):
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        scores = dets[:, 4]
        keep = []
        index = scores.argsort()[::-1]

        while index.size > 0:
            i = index[0]
            keep.append(i)
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)

            overlaps = w * h
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            idx = np.where(ious <= thresh)[0]
            index = index[idx + 1]
        return keep

    @staticmethod
    def filter_box(org_box, conf_thres, iou_thres):
        org_box = np.squeeze(org_box)
        conf = org_box[..., 4] > conf_thres
        box = org_box[conf == True]

        cls_cinf = box[..., 5:]
        cls = [int(np.argmax(x)) for x in cls_cinf]
        all_cls = list(set(cls))

        out = []
        for curr_cls in all_cls:
            curr_cls_box = []
            for j in range(len(cls)):
                if cls[j] == curr_cls:
                    box[j][5] = curr_cls
                    curr_cls_box.append(box[j][:6])
            curr_cls_box = np.array(curr_cls_box)
            curr_cls_box = transTool.xywh2xyxy(curr_cls_box)
            curr_out_box = transTool.nms(curr_cls_box, iou_thres)
            out.extend(curr_cls_box[k] for k in curr_out_box)
        return np.array(out)

class onnxDetect():
    def __init__(self):
        self.model = YOLOV5('/home/pi/四足机器人案例123/动物识别/best.onnx')

    def runDetection(self, image):
        try:
            output, processed_img = self.model.inference(image)
            outbox = transTool.filter_box(output, 0.5, 0.5)
            
            if len(outbox) == 0:
                return None, None, processed_img
                
            scores, classes, boxes = outbox[..., 4], outbox[..., 5].astype(np.int32), outbox[..., :4]

            # Ensure image is in correct format for OpenCV
            if processed_img.dtype != np.uint8:
                processed_img = (processed_img * 255).astype(np.uint8)
            if len(processed_img.shape) == 3 and processed_img.shape[2] == 3:
                pass  # Already in BGR format
            else:
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

            for box, score, cl in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(processed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_img, f'{CLASSES[cl]}: {score:.2f}', 
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return CLASSES[classes[0]], boxes, processed_img
        except Exception as e:
            print(f"Error in detection: {str(e)}")
            return None, None, image
#******************************************************************************************************************************************#


class DisasterDetector:
    def __init__(self, onnx_path):
        """初始化检测器"""
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self._get_input_name()
        self.output_name = self._get_output_name()
    
    def _get_input_name(self):
        return [node.name for node in self.onnx_session.get_inputs()]
    
    def _get_output_name(self):
        return [node.name for node in self.onnx_session.get_outputs()]
    
    def _preprocess(self, image):
        """图像预处理"""
        img = cv2.resize(image, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.stack([img]*8, axis=0)  # Shape [8, 3, 640, 640]
    
    def _postprocess(self, pred):
        """后处理检测结果"""
        # 简化的后处理逻辑，只返回最高置信度的类别
        pred = np.squeeze(pred)
        conf = pred[..., 4] > 0.5  # 置信度阈值
        box = pred[conf == True]
        
        if len(box) == 0:
            return None
            
        # 获取最高置信度的类别
        cls_conf = box[..., 5:]
        cls_idx = np.argmax(cls_conf, axis=1)
        max_idx = np.argmax(box[:, 4])
        
        return CLASSES[cls_idx[max_idx]], float(box[max_idx, 4])
    
    def detect(self, image):
        """
        对输入图像进行灾害检测
        :param image: numpy数组格式的BGR图像
        :return: (灾害类别, 置信度) 或 None(未检测到)
        """
        try:
            # 预处理
            input_tensor = self._preprocess(image)
            input_feed = {name: input_tensor for name in self.input_name}
            
            # 推理
            pred = self.onnx_session.run(self.output_name, input_feed)[0]
            
            # 后处理
            return self._postprocess(pred)
        except Exception as e:
            print(f"检测错误: {str(e)}")
            return None

#******************************************************************************************************************************************#



if __name__ == '__main__':
    model = onnxDetect()
    # XGO_edu.xgoCamera(True)
    # XGO_edu.xgoTakePhoto(filename="captured_image.jpg")
    image = cv2.imread('/home/pi/xgoPictures/photo.jpg')
    
    if image is None:
        print("Error: Could not load image")
    else:
        result, box, img = model.runDetection(image)
        
        if result:
            print(f"Detection result: {result}")
            # cv2.imshow('Detection Result', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite('/home/pi/result.jpg', img)
            print("结果已保存到 /home/pi/result.jpg")
        else:
            print("No objects detected")