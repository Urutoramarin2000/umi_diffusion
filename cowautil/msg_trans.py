import numpy as np
from cowautil.msg_communicator import MsgSubscriber, MsgPublisher
from cowautil.realsense.realsense_tools import RealSenseTools
import threading
import cv2
import time
import base64

# tbox password:B7H0farqbKM12eHj8c0FF0wwDlfCgOQK
class MsgTrans:

    def __init__(self):
        self.d405_img = None
        self.d415_img = None

    def run_arm_action_trans(self):
        arm_action_sub = MsgSubscriber(ip='192.168.2.30', port=22222, topic='arm_action')
        arm_action_pub = MsgPublisher(port=22222)
        while True:
            msg = arm_action_sub.recv()
            print(msg)
            arm_action_pub.send(topic='arm_action', msg=msg)

    def run_camera_d405_trans(self):
        camera_d405_sub = RealSenseTools(cameral_serial='218622275838')
        camera_d405_pub = MsgPublisher(port=22405)
        while True:
            color_img, depth_img, depth_frame, depth_colormap = camera_d405_sub.recv()
            self.d405_img = color_img
            msg = {'data': base64.b64encode(color_img.tobytes()).decode('utf-8'), 'shape': color_img.shape}
            camera_d405_pub.send(topic='camera_d405', msg=msg)
            print('d405sent')


    def run_camera_d415_trans(self):
        camera_d415_sub = RealSenseTools(cameral_serial='748512060307')
        camera_d415_pub = MsgPublisher(port=22415)
        while True:
            color_img, depth_img, depth_frame, depth_colormap = camera_d415_sub.recv()
            self.d415_img = color_img
            msg = {'data': base64.b64encode(color_img.tobytes()).decode('utf-8'), 'shape': color_img.shape}
            camera_d415_pub.send(topic='camera_d415', msg=msg)
            print('d415sent')

            

    def run(self):
        thread_list = []
        thread_list.append(threading.Thread(target=self.run_arm_action_trans))
        thread_list.append(threading.Thread(target=self.run_camera_d405_trans))
        thread_list.append(threading.Thread(target=self.run_camera_d415_trans))
        for th in thread_list:
            th.start()
        while True:
            if self.d415_img is None or self.d405_img is None:
                time.sleep(0.1)
                continue
            cv2.imshow('Viewer', np.concatenate([self.d405_img, self.d415_img], axis=1))
            if cv2.waitKey(10) == ord('q'):
                break
        for th in thread_list:
            th.join()

if __name__ == '__main__':
    msg_trans = MsgTrans()
    msg_trans.run()