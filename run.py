#!/usr/bin/env python
import numpy as np
import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.flow_utils import visualize_flow_file
from utils.load_data import *

# check if GPU is available
print('Cuda is available:', torch.cuda.is_available(),
      'Using', torch.cuda.get_device_name(torch.cuda.current_device()))

try:
    from .correlation import correlation  # the custom cost volume layer
except:
    sys.path.insert(0, './correlation')
    import correlation  # you should consider upgrading python
# end

##########################################################

assert (int(str('').join(torch.__version__.split('.')[0:2])) >= 13)  # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'kitti'  # 'default', or 'kitti', or 'sintel'
# arguments_strOne = './images/frame02.png'
# arguments_strTwo = './images/frame01.png'
# arguments_strOut = './out_backward.flo'

dataset_dir = '/dataset/'
dataset_name = 'kitti_odom_optflo/'
img_seq = 'training/'
cam = 'image_2/'
arguments_strImgDir = dataset_dir + dataset_name + img_seq + cam
arguments_strImgOut = "/results/" + dataset_name + img_seq

for strOption, strArgument in \
        getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument  # which model to use
    if strOption == '--one' and strArgument != '': arguments_strOne = strArgument  # path to the first frame
    if strOption == '--two' and strArgument != '': arguments_strTwo = strArgument  # path to the second frame
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument  # path to where the output should be stored
# end

##########################################################

backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(
            1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(
            1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()
    # end

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenInput,
                                           grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1),
                                           mode='bilinear', padding_mode='zeros', align_corners=False)


# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Features(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            # end

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)

                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]
        # end

        # end

        class Matching(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                self.fltBackwarp = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]

                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()

                elif intLevel == 2:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )

                # end

                if intLevel == 6:
                    self.netUpflow = None

                elif intLevel != 6:
                    self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2,
                                                              padding=1, bias=False, groups=2)

                # end

                if intLevel >= 4:
                    self.netUpcorr = None

                elif intLevel < 4:
                    self.netUpcorr = torch.nn.ConvTranspose2d(in_channels=49, out_channels=49, kernel_size=4, stride=2,
                                                              padding=1, bias=False, groups=49)

                # end

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel],
                                    stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel])
                )

            # end

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenFeaturesOne = self.netFeat(tenFeaturesOne)
                tenFeaturesTwo = self.netFeat(tenFeaturesTwo)

                if tenFlow is not None:
                    tenFlow = self.netUpflow(tenFlow)
                # end

                if tenFlow is not None:
                    tenFeaturesTwo = backwarp(tenInput=tenFeaturesTwo, tenFlow=tenFlow * self.fltBackwarp)
                # end

                if self.netUpcorr is None:
                    tenCorrelation = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(tenOne=tenFeaturesOne, tenTwo=tenFeaturesTwo,
                                                              intStride=1), negative_slope=0.1, inplace=False)

                elif self.netUpcorr is not None:
                    tenCorrelation = self.netUpcorr(torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(tenOne=tenFeaturesOne, tenTwo=tenFeaturesTwo,
                                                              intStride=2), negative_slope=0.1, inplace=False))

                # end

                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(tenCorrelation)
        # end

        # end

        class Subpixel(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]

                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()

                elif intLevel == 2:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )

                # end

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[0, 0, 130, 130, 194, 258, 386][intLevel], out_channels=128,
                                    kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel],
                                    stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel])
                )

            # end

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenFeaturesOne = self.netFeat(tenFeaturesOne)
                tenFeaturesTwo = self.netFeat(tenFeaturesTwo)

                if tenFlow is not None:
                    tenFeaturesTwo = backwarp(tenInput=tenFeaturesTwo, tenFlow=tenFlow * self.fltBackward)
                # end

                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(
                    torch.cat([tenFeaturesOne, tenFeaturesTwo, tenFlow], 1))
        # end

        # end

        class Regularization(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]

                self.intUnfold = [0, 0, 7, 5, 5, 3, 3][intLevel]

                if intLevel >= 5:
                    self.netFeat = torch.nn.Sequential()

                elif intLevel < 5:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=[0, 0, 32, 64, 96, 128, 192][intLevel], out_channels=128,
                                        kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )

                # end

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[0, 0, 131, 131, 131, 131, 195][intLevel], out_channels=128,
                                    kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                if intLevel >= 5:
                    self.netDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                                        kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1,
                                        padding=[0, 0, 3, 2, 2, 1, 1][intLevel])
                    )

                elif intLevel < 5:
                    self.netDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                                        kernel_size=([0, 0, 7, 5, 5, 3, 3][intLevel], 1), stride=1,
                                        padding=([0, 0, 3, 2, 2, 1, 1][intLevel], 0)),
                        torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                                        out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                                        kernel_size=(1, [0, 0, 7, 5, 5, 3, 3][intLevel]), stride=1,
                                        padding=(0, [0, 0, 3, 2, 2, 1, 1][intLevel]))
                    )

                # end

                self.netScaleX = torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=1,
                                                 kernel_size=1, stride=1, padding=0)
                self.netScaleY = torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=1,
                                                 kernel_size=1, stride=1, padding=0)

            # eny

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenDifference = (tenOne - backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackward)).square().sum(1,
                                                                                                                      True).sqrt().detach()

                tenDist = self.netDist(self.netMain(torch.cat([tenDifference,
                                                               tenFlow - tenFlow.view(tenFlow.shape[0], 2, -1).mean(2,
                                                                                                                    True).view(
                                                                   tenFlow.shape[0], 2, 1, 1),
                                                               self.netFeat(tenFeaturesOne)], 1)))
                tenDist = tenDist.square().neg()
                tenDist = (tenDist - tenDist.max(1, True)[0]).exp()

                tenDivisor = tenDist.sum(1, True).reciprocal()

                tenScaleX = self.netScaleX(
                    tenDist * torch.nn.functional.unfold(input=tenFlow[:, 0:1, :, :], kernel_size=self.intUnfold,
                                                         stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(
                        tenDist)) * tenDivisor
                tenScaleY = self.netScaleY(
                    tenDist * torch.nn.functional.unfold(input=tenFlow[:, 1:2, :, :], kernel_size=self.intUnfold,
                                                         stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(
                        tenDist)) * tenDivisor

                return torch.cat([tenScaleX, tenScaleY], 1)
        # end

        # end

        self.netFeatures = Features()
        self.netMatching = torch.nn.ModuleList([Matching(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.netSubpixel = torch.nn.ModuleList([Subpixel(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.netRegularization = torch.nn.ModuleList([Regularization(intLevel) for intLevel in [2, 3, 4, 5, 6]])

        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                              torch.hub.load_state_dict_from_url(
                                  url='http://content.sniklaus.com/github/pytorch-liteflownet/network-' + arguments_strModel + '.pytorch',
                                  file_name='liteflownet-' + arguments_strModel).items()})

    # end

    def forward(self, tenOne, tenTwo):
        tenOne[:, 0, :, :] = tenOne[:, 0, :, :] - 0.411618
        tenOne[:, 1, :, :] = tenOne[:, 1, :, :] - 0.434631
        tenOne[:, 2, :, :] = tenOne[:, 2, :, :] - 0.454253

        tenTwo[:, 0, :, :] = tenTwo[:, 0, :, :] - 0.410782
        tenTwo[:, 1, :, :] = tenTwo[:, 1, :, :] - 0.433645
        tenTwo[:, 2, :, :] = tenTwo[:, 2, :, :] - 0.452793

        tenFeaturesOne = self.netFeatures(tenOne)
        tenFeaturesTwo = self.netFeatures(tenTwo)

        tenOne = [tenOne]
        tenTwo = [tenTwo]

        for intLevel in [1, 2, 3, 4, 5]:
            tenOne.append(torch.nn.functional.interpolate(input=tenOne[-1], size=(
                tenFeaturesOne[intLevel].shape[2], tenFeaturesOne[intLevel].shape[3]), mode='bilinear',
                                                          align_corners=False))
            tenTwo.append(torch.nn.functional.interpolate(input=tenTwo[-1], size=(
                tenFeaturesTwo[intLevel].shape[2], tenFeaturesTwo[intLevel].shape[3]), mode='bilinear',
                                                          align_corners=False))
        # end

        tenFlow = None

        for intLevel in [-1, -2, -3, -4, -5]:
            tenFlow = self.netMatching[intLevel](tenOne[intLevel], tenTwo[intLevel], tenFeaturesOne[intLevel],
                                                 tenFeaturesTwo[intLevel], tenFlow)
            tenFlow = self.netSubpixel[intLevel](tenOne[intLevel], tenTwo[intLevel], tenFeaturesOne[intLevel],
                                                 tenFeaturesTwo[intLevel], tenFlow)
            tenFlow = self.netRegularization[intLevel](tenOne[intLevel], tenTwo[intLevel], tenFeaturesOne[intLevel],
                                                       tenFeaturesTwo[intLevel], tenFlow)
        # end

        return tenFlow * 20.0
# end


# end

netNetwork = None


##########################################################

def estimate(tenOne, tenTwo):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()
    # end

    assert (tenOne.shape[1] == tenTwo.shape[1])
    assert (tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    assert (
            intWidth == 1024)  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert (
            intHeight == 436)  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne,
                                                         size=(intPreprocessedHeight, intPreprocessedWidth),
                                                         mode='bilinear', align_corners=False)
    tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo,
                                                         size=(intPreprocessedHeight, intPreprocessedWidth),
                                                         mode='bilinear', align_corners=False)

    tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo),
                                              size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu()


# end

##########################################################


if __name__ == '__main__':
    # estimation on RGB image
    # tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    # tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strTwo))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

    # estimation on duplicated gray image
    # tenOne_gray = PIL.Image.open(arguments_strOne).resize((1024, 436))
    # tenTwo_gray = PIL.Image.open(arguments_strTwo).resize((1024, 436))
    # tenOne = np.zeros((436, 1024, 3))
    # tenOne[:, :, 0] = tenOne_gray
    # tenOne[:, :, 1] = tenOne_gray
    # tenOne[:, :, 2] = tenOne_gray
    #
    # tenTwo = np.zeros((436, 1024, 3))
    # tenTwo[:, :, 0] = tenTwo_gray
    # tenTwo[:, :, 1] = tenTwo_gray
    # tenTwo[:, :, 2] = tenTwo_gray

    # estimation on sequence
    timestamp = 200

    sys.stdout.write("[%s]" % (" " * timestamp))
    sys.stdout.flush()
    sys.stdout.write("\b" * (timestamp + 1))

    timer = 0.0

    for i in range(timestamp):
        # tenOne = get_image(arguments_strImgDir, i)
        # tenTwo = get_image(arguments_strImgDir, i + 1)
        tenOne, size_one = get_image_kitti_flow(arguments_strImgDir, 0, i)
        tenTwo, size_two = get_image_kitti_flow(arguments_strImgDir, 1, i)

        start_time = time.time()

        tenOne = torch.FloatTensor(
            numpy.ascontiguousarray(tenOne.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
        tenTwo = torch.FloatTensor(
            numpy.ascontiguousarray(tenTwo.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

        tenOutput = estimate(tenOne, tenTwo)

        timer += time.time() - start_time

        out_path_flo = os.path.dirname(os.path.abspath(__file__)) + arguments_strImgOut + "flo/"
        make_dir_if_not_exist(out_path_flo)

        out_file = os.path.join(out_path_flo, "{:06d}.{}".format(i, 'flo'))

        objOutput = open(out_file, 'wb')

        numpy.array([80, 73, 69, 72], numpy.uint8).tofile(objOutput)
        numpy.array([tenOutput.shape[2], tenOutput.shape[1]], numpy.int32).tofile(objOutput)
        numpy.array(tenOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)

        objOutput.close()

        # visualize resutls
        vis_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)) + arguments_strImgOut, "vis/")
        make_dir_if_not_exist(vis_save_dir)
        visualize_flow_file(out_file, save_dir=vis_save_dir, resize=True, format='kitti', origin_size=size_one)

        progress = i / (timestamp - 1)
        block = int(round(100 * progress))
        text = "\rProcess: [{0}] {1}%".format("#" * block + "-" * (100 - block), round(progress * 100))
        sys.stdout.write(text)
        sys.stdout.flush()

    # generate video of results
    # generate_video = True
    # if generate_video:
    #     generate_video_compare(arguments_strImgDir, arguments_strImgOut, timestamp)

    sys.stdout.write(" Done!\n")
    print("Total time for estimation: %lf" % timer)
    print("Average time for estimating one frame %lf" % (timer / timestamp))

# end
