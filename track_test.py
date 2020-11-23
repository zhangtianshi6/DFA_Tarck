from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import _init_paths
import os
import os.path as osp
import cv2
import argparse
import motmetrics as mm
import numpy as np
import torch

from multitracker import JDETracker


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh

def load_img(img0):
        img_size=(1088, 608)
        img, _, _, _ = letterbox(img0, height=img_size[1], width=img_size[0])
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img, img0

def main():
    v_path = "/home/kesci/input/test3454/"
    v_ls = os.listdir(v_path)
    min_box_area = 0.3
    load_model_name = "pt/model_track.pth"
    #gpus = '0, 1'
    #os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    gpus = ""
    frame_rate=30
    for i in range(len(v_ls)):
        tracker = JDETracker(gpus, load_model_name, frame_rate=frame_rate)
        results = []
        frame_id = 0
        # if i==0:
        #     v_ls[i] = "uav0000013_00000_v_test.mp4"
        video_path = v_path+v_ls[i]
        cap = cv2.VideoCapture(video_path)
        ret, img0 = cap.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_path = '/home/kesci/work/video/'+ v_ls[i].split('.')[0]+'.avi'
        save_txt = '/home/kesci/work/video/'+ v_ls[i].split('.')[0]+'.txt'
        fw = open(save_txt, 'w')
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        out = cv2.VideoWriter(save_path,fourcc, 20.0, (img0.shape[1],img0.shape[0]))
        print("read video", video_path)
        while(ret):
            ret, img0 = cap.read()
            try:
            #if True:
                shape = img0.shape
                img,img0 = load_img(img0)
                # run tracking
                #blob = torch.from_numpy(img).cuda().unsqueeze(0)
                blob = torch.from_numpy(img).unsqueeze(0)
                online_targets = tracker.update(blob, img0)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                # save results
                results.append((frame_id + 1, online_tlwhs, online_ids))
                for tlwh, track_id in zip(online_tlwhs, online_ids):
                        if track_id < 0:
                            continue
                        x1, y1, w, h = tlwh
                        x2, y2 = x1 + w, y1 + h
                        #print('box', x1,y1,x2,y2)
                        cv2.rectangle(img0, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                        line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                        fw.write(line)
                        cv2.putText(img0, str(track_id), (int(x1),int(y1)), font, 1.2, (255, 255, 255), 2)
                out.write(img0)
                frame_id += 1
            except:
                print(ret, video_path)
        print("save path is", save_path)
            
    return 0


if __name__ == '__main__':
    main()
