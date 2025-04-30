import zarr
import argparse
import cv2
from imagecodecs_numcodecs import register_codecs
from msg_communicator import MsgPublisher

register_codecs()

def replay(args):
    cmd_pub = MsgPublisher(port=22222)
    dataset = zarr.open(args.record_file, mode='r')
    for i in range(len(dataset['/data/camera0_rgb'])):
        # memory = dataset['/data/memory'][i]
        img = dataset['/data/camera0_rgb'][i]
        rot = dataset['/data/ee_rot'][i]
        gripper_width = dataset['/data/gripper_pos'][i]
        # info_text = f"Index: {i}, Memory: {memory}"
        info_text2 = f"gripper:{gripper_width}"
        info_text3 = f"rot:{rot}"
        # cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(img, info_text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(img, info_text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        # print(memory)
        cv2.imshow('replay', img)
        if cv2.waitKey(0) == ord('q'):
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_file', '-f', type=str,
                        help='Path to the record file')
    args = parser.parse_args()

    replay(args)