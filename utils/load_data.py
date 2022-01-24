import PIL.Image
import numpy as np
import cv2
import glob
import sys
import os

img_ext = 'png'


def make_dir_if_not_exist(path):
    """ Make a directory if the input path does not exist
    Args:
        path (str): path to check and eventually create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_image(img_dir, timestamp):
    """ Get image data given timestamp
    Args:
        img_dir (str): image directory
        timestamp (int): timestamp for the image

    Returns:
        im (np.array, [436, 1024, 3]): image data
    """
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + img_dir,
                            "{:06d}.{}".format(timestamp, img_ext))
    img = PIL.Image.open(img_path)
    img = img.resize((1024, 436))
    # img.show()

    img_array = np.array(img)
    if len(img_array.shape) > 2:  # RGB
        im = img_array[:, :, ::-1]

    else:                         # GRAY
        gray = np.array(img)
        # print(gray.shape)
        im = np.zeros((436, 1024, 3))
        im[:, :, 0] = gray
        im[:, :, 1] = gray
        im[:, :, 2] = gray

    return im


def get_image_euroc(img_path):
    """ Get image of EuRoC dataset, duplicate it in three channels and resize it to 436 x 1024
    Args:
        img_path (str): absolute path to the image data
    Returns:
        im (np.array, [436, 1024, 3]): image with three channels in 436 x 1024
        origin_size (tuple): (height, width) of the original image data
    """
    img = PIL.Image.open(img_path)
    origin_size = img.size
    # img = img.resize((1024, 436))
    gray = np.array(img)
    im = np.zeros((img.size[1], img.size[0], 3))
    im[:, :, 0] = gray
    im[:, :, 1] = gray
    im[:, :, 2] = gray

    return im, origin_size


def get_image_kitti_flow(img_dir, frame, timestamp):
    """ Load image data of KITTI Flow 2015 dataset (RGB) and resize it to 436 x 1024
    Args:
        img_dir (str): relative path to the folder containing the image data
        frame (int): 0 for the first image and 1 for the second image
        timestamp (int): timestamp of the image to load
    Returns:
        im (np.array, [1024, 436, 3]) BGR image in numpy array
        origin_size (tuple): (height, width) of the original image data
    """
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + img_dir,
                            "{:06d}_1{}.{}".format(timestamp, frame, img_ext))
    img = PIL.Image.open(img_path)
    origin_size = img.size
    img = img.resize((1024, 436))
    im = np.array(img)[:, :, ::-1]

    return im, origin_size


def generate_video_raw(raw_dir, res_path, timestamp):
    frame_size = (1024, 436)
    out = cv2.VideoWriter(res_path + 'videos/origin_video.avi', cv2.VideoWriter_fourcc(*'mp4v'), 60,
                          frame_size)

    for i in range(timestamp + 1):

        img = np.uint8(get_image(raw_dir, i))
        out.write(img)

    out.release()
    cv2.destroyAllWindows()


def generate_video_result(results_dir, res_path, timestamp):
    frame_size = (1024, 436)
    out = cv2.VideoWriter(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + res_path + 'result_video.avi',
                          cv2.VideoWriter_fourcc(*'mp4v'), 60, frame_size)
    for i in range(timestamp):
        img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + results_dir,
                                "{:06d}_10.{}".format(i, img_ext))
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        cv2.imshow('frame', img)
        out.write(img)

    out.release()


def generate_video_compare(raw_dir, results_dir):
    raw_dir_abs = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + raw_dir
    res_dir_abs = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + results_dir
    raw_paths = sorted(glob.glob(raw_dir_abs + '*.png'))
    res_paths = sorted(glob.glob(res_dir_abs + '*.png'))

    raw_height, raw_width, raw_channel = cv2.imread(raw_paths[0]).shape
    res_height, res_width, res_channel = cv2.imread(res_paths[0]).shape

    assert (raw_width == res_width) & (raw_height == res_height)

    make_dir_if_not_exist(res_dir_abs + 'videos/')
    out = cv2.VideoWriter(res_dir_abs + 'videos/result_video.avi',
                          cv2.VideoWriter_fourcc(*'mp4v'), 60, (raw_width * 2, raw_height))

    for i in range(len(raw_paths) - 1):

        # img_raw = np.uint8(get_image(raw_dir, i))
        img_raw = cv2.imread(raw_paths[i])
        img_res = cv2.imread(res_paths[i], cv2.IMREAD_UNCHANGED)

        both = np.concatenate((img_raw, img_res), axis=1)

        cv2.imshow('Frame', both)
        out.write(both)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Test


# generate_video_result("/results/kitti_odom_gray_00/vis/", 4540)
# generate_video_raw("/dataset/kitti_odom_gray/00/image_0/", 4540)
# generate_video_compare("/dataset/kitti_odom_gray/10/image_0/", "/results/kitti_odom_gray_10/vis/", 1200)
# img = get_image('/dataset/kitti_odom_color/00/image_2/', 6)
# cv2.imshow("frame", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# generate_video_result('/results/evaluate01/data/flow/', '/results/evaluate01/', 200)
# generate_video_compare('/dataset/kitti_odom_optflo/training/flow_noc/', '/results/evaluate01/data/flow/', 200)
# get_image_euroc('/home/jingkun/SemesterProject/pytorch-liteflownet/dataset/euroc/MH_04_difficult/mav0/cam0/data/1403638127245096960.png')
# generate_video_compare('/dataset/euroc/MH_04_difficult/mav0/cam0/data/', '/results/euroc/MH_04_difficult/mav0/vis/')
