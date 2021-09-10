import PIL.Image
import numpy as np
import cv2
import glob
import sys
import os

img_ext = 'png'


def get_image(img_dir, timestamp):
    """ Get image data given timestamp

    Args:
        img_dir (str): image directory
        timestamp (int): timestamp for the image

    Returns:
        img (np.array, [436, 1024, 3]): image data
    """
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + img_dir,
                            "{:06d}.{}".format(timestamp, img_ext))
    img = PIL.Image.open(img_path)
    img = img.resize((1024, 436))
    # img.show()

    gray = np.array(img)
    # print(gray.shape)
    im = np.zeros((436, 1024, 3))
    im[:, :, 0] = gray
    im[:, :, 1] = gray
    im[:, :, 2] = gray

    return im


def generate_video_raw(raw_dir, timestamp):
    frame_size = (1024, 436)
    out = cv2.VideoWriter('../results/kitti_odom_gray_00/videos/origin_video.avi', cv2.VideoWriter_fourcc(*'mp4v'), 60,
                          frame_size)

    for i in range(timestamp + 1):

        img = np.uint8(get_image(raw_dir, i))
        out.write(img)

    out.release()
    cv2.destroyAllWindows()


def generate_video_result(results_dir, timestamp):
    frame_size = (1024, 436)
    out = cv2.VideoWriter('../results/kitti_odom_gray_00/videos/result_video.avi', cv2.VideoWriter_fourcc(*'mp4v'), 60,
                          frame_size)
    for i in range(timestamp + 1):
        img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + results_dir,
                                "{:06d}-vis.{}".format(i, img_ext))
        img = cv2.imread(img_path)
        out.write(img)

    out.release()

def generate_video_compare(raw_dir, results_dir, timestamp):
    frame_size = (1024 * 2, 436)

    out = cv2.VideoWriter('../results/kitti_odom_gray_10/videos/result_video.avi', cv2.VideoWriter_fourcc(*'mp4v'), 60,
                          frame_size)

    for i in range(timestamp):

        img_raw = np.uint8(get_image(raw_dir, i))
        img_res = cv2.imread(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + results_dir,
                                "{:06d}-vis.{}".format(i, img_ext)))

        both = np.concatenate((img_raw, img_res), axis=1)

        cv2.imshow('Frame', both)
        out.write(both)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#
# generate_video_result("/results/kitti_odom_gray_00/vis/", 4540)
# generate_video_raw("/dataset/kitti_odom_gray/00/image_0/", 4540)
generate_video_compare("/dataset/kitti_odom_gray/10/image_0/", "/results/kitti_odom_gray_10/vis/", 1200)
