import numpy as np
import matplotlib.pyplot as plt
import os.path
import cv2

TAG_CHAR = np.array([202021.25], np.float32)


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


# ref: https://github.com/sampepose/flownet2-tf/
# blob/18f87081db44939414fc4a48834f9e0da3e69f4c/src/flowlib.py#L240
# modified accordingly based on the project
def visualize_flow_file(flow_filename, save_dir=None, format='middlebury', resize=False, origin_size=None):
    # print(flow_filename)  # Debug
    flow_data = readFlow(flow_filename)

    if format == 'middlebury':
        img = flow2img(flow_data)

        if resize & (origin_size is not None):
            img = cv2.resize(img, origin_size)

    elif format == 'kitti':
        # print(flow_data.shape[1], '/', origin_size[0])
        # print(flow_data.shape[0], '/', origin_size[1])
        flow_data[:, :, 0] /= float(flow_data.shape[1] / origin_size[0])
        flow_data[:, :, 1] /= float(flow_data.shape[0] / origin_size[1])
        img = flow_to_png_kitti(flow_data)

        if resize & (origin_size is not None):
            img = cv2.resize(img, origin_size)

    if save_dir:
        idx = flow_filename.rfind("/") + 1
        # plt.imsave(os.path.join(save_dir, "%s_10.png" % flow_filename[idx:-4]), img)
        # cv2.imwrite(os.path.join(save_dir, "%s_10.png" % flow_filename[idx:-4]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "%s.png" % flow_filename[idx:-4]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def flow2img(flow_data):
    """
	convert optical flow into color image
	:param flow_data:
	:return: color image
	"""
    # print(flow_data.shape)
    # print(type(flow_data))
    u = flow_data[:, :, 0]
    v = flow_data[:, :, 1]

    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
	compute optical flow color map
	:param u: horizontal optical flow
	:param v: vertical optical flow
	:return:
	"""

    height, width = u.shape
    img = np.zeros((height, width, 3))

    NAN_idx = np.isnan(u) | np.isnan(v)
    u[NAN_idx] = v[NAN_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

    return img


def make_color_wheel():
    """
	Generate color wheel according Middlebury color code
	:return: Color wheel
	"""
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def flow_to_png_kitti(flow_data):
    """ Convert flo file to png file required by KITTI evaluation

    Arg:
        flow (np.array [h, w, 2]): 2-band float image for horizontal (u) and vertical (v) flow components

    Returns:
        flow_image (np.array [h, w, 3]): optical flow maps as 3-channel uint16 array with the first channel contains
        u-component, the second channel the v-component and the third channel denotes if the pixel is valid or not (1 if
        true, 0 otherwise).
    """
    flow_image = np.zeros((flow_data.shape[0], flow_data.shape[1], 3), dtype=np.uint16)
    # flow[:, :, 0] = flow[:, :, 0].astype(np.float32) * float(64.0) + float(32768.0)
    # flow[:, :, 1] = flow[:, :, 1].astype(np.float32) * float(64.0) + float(32768.0)
    # valid = is_valid(flow[:, :, 0], flow[:, :, 1])
    u = flow_data[:, :, 0]
    v = flow_data[:, :, 1]
    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)

    flow_image[:, :, 2] = is_valid(u, v).astype(np.uint16)
    flow_image[:, :, 2][idx_unknown] = 0

    flow_image[:, :, 0] = u.astype(np.float32) * float(64.0) + float(32768.0)
    flow_image[:, :, 1] = v.astype(np.float32) * float(64.0) + float(32768.0)
    return flow_image


def is_valid(u, v):
    """ Check if optical flow at given pixel is valid

    Arg:
        u (np.array [h, w, 1]): u-component of optical flow
        v (np.array [h, w, 1]): v-component of optical flow

    Returns:
        valid (np.array [h, w, 1]): denotes if the pixel is valid or not
    """
    assert u.shape == v.shape
    height, width,  = u.shape
    valid = np.ones((height, width))

    return valid


def uv_flow_to_float(flow_data):
    u = flow_data[:, :, 0]
    v = flow_data[:, :, 1]
    flow_float = np.zeros((u.shape[0], u.shape[1], 2), dtype=np.float32)
    flow_float[:, :, 0] = (u.astype(np.float64) - float(32768.0)) / float(64.0)
    flow_float[:, :, 1] = (v.astype(np.float64) - float(32768.0)) / float(64.0)

    return flow_float


# fn = readFlow("../test.flo")
# flow_uint16 = flow_to_png_kitti(fn)
# flow_float = uv_flow_to_float(flow_uint16)
# print(flow_float.dtype)
# cv2.imshow('uint16 map', flow_uint16[:, :, ::-1])
# img = flow2img(flow_float)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# cv2.imshow('Color_map', img)


# visualize_flow_file("/home/jingkun/SemesterProject/pytorch-liteflownet/results/kitti_odom_optflo/training/flo/000000.flo",
#                     save_dir='../', format='kitti', resize=True,
#                     origin_size=(1242, 375))
# flow_uint16 = cv2.imread('../000000_10.png', cv2.IMREAD_UNCHANGED)
# print(flow_uint16[:, :, 0])
# cv2.imshow("flow", flow_uint16)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# flow_data = readFlow("/home/jingkun/SemesterProject/pytorch-liteflownet/results/euroc/MH_05_difficult/mav0/cam0/flo/1403638518077829376.flo")
# flow_data = readFlow('/home/jingkun/SemesterProject/pytorch-liteflownet/images/flo/frame01.flo')
# u = np.array(flow_data[:, :, 0])
# v = np.array(flow_data[:, :, 1])
# print(u.shape)
# # To normalize the flow data
# flow = np.sqrt(u ** 2 + v ** 2)
# normalized_flow = flow / np.max(flow)
# normalized_flow *= 255
# print(normalized_flow)
# cv2.imshow('normalized flow', normalized_flow.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import glob
#
# path = '../images/flo/' + '*.flo'
# flo_path = sorted(glob.glob(path))
#
# for flo in flo_path:
#     flo_mat = cv2.readOpticalFlow(flo)
#     u = np.array(flo_mat[:, :, 0])
#     v = np.array(flo_mat[:, :, 1])
#     print(u)
#     print(v)
#     print(u.shape)
#     flow = np.sqrt(u ** 2 + v ** 2)
#     normalized_flow = flow / np.max(flow)
#     normalized_flow *= 255
#     print(normalized_flow)
#     cv2.imshow('normalized flow', normalized_flow.astype(np.uint8))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# path_1 = '../images/flo/000486_lr.flo'
# path_2 = '../images/flo/000486_rl.flo'
#
# flow_1_data = cv2.readOpticalFlow(path_1)
# flow_u1 = np.array(flow_1_data[:, :, 0])
# flow_v1 = np.array(flow_1_data[:, :, 1])
# flow_2_data = cv2.readOpticalFlow(path_2)
# flow_u2 = np.array(flow_2_data[:, :, 0])
# flow_v2 = np.array(flow_2_data[:, :, 1])
# row, col, _ = flow_1_data.shape
# val_flo = 0
# val_track = 0
# reproject_error_2 = 0
# for r in range(row):
#     for c in range(col):
#         u = flow_u1[r, c]
#         v = flow_v1[r, c]
#         if (r + v < row) & (c + u < col) & (r + v > 0) & (c + u > 0):
#             val_flo += 1
#             inv_u = flow_u2[int(r + v), int(c + u)]
#             inv_v = flow_v2[int(r + v), int(c + u)]
#             print(u, v, inv_u, inv_v)
#             dist = (u + inv_u) ** 2 + (v + inv_v) ** 2
#             reproject_error_2 += dist
#             if dist < 0.25:
#                 val_track += 1
#
# print(val_flo / (row * col))
# print(val_track / (row * col))
# print(reproject_error_2 / val_flo)
#
# flow_u1 = np.zeros(flow_u1.shape, dtype=flow_u1.dtype)
# flow_v1 = np.zeros(flow_v1.shape, dtype=flow_v1.dtype)
# print(flow_u1, flow_v1)
# writeFlow('./000000.flo', flow_u1, flow_v1)


# u2 = np.array(flow_2_data[:, :, 0])[148 + int(v1), 81 + int(u1)]
# v2 = np.array(flow_2_data[:, :, 1])[148 + int(v1), 81 + int(u1)]
# print('{0:.6f}, {1:.6f}'.format(u1, v1), '{0:.6f}, {1:.6f}'.format(u2, v2))
# print(((u1 - u2) ** 2 + (v1 - v2) ** 2) < 0.04)
