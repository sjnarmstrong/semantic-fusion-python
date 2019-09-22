from SemanticFusion.cnn_interface.pyCaffeInterface import CaffeInterface
from SemanticFusion.semantic_fusion.pySemanticFusionInterface import SemanticFusionInterface
from SemanticFusion.map_interface.pyElasticFusionInterface import Resolution, Intrinsics, ElasticFusionInterface
from SemanticFusion.utilities.pyLogReader import LogReader, PNGLogReader
from SemanticFusion.utilities.pyTypes import ClassColour, VectorOfClassColour
from SemanticFusion.gui.pyGui import Gui
import numpy as np

from PIL import Image

# def load_colour_scheme(filename):
#     with open(filename) as fp:
#         data = fp.read().split('\n')[2:]
#     class_names, class_ids, rs, gs, bs, = zip(*(vals.split()[:5] for vals in data[:-1]))
#
#     class_ids = np.array(class_ids, dtype=np.uint16)
#     number_of_classes = class_ids.max()+1
#     colormap = np.empty((number_of_classes, 3), dtype=np.uint8)
#     colormap.T[:, class_ids] = rs, gs, bs
#     # print(class_names)
#     # print(colormap)
#     return colormap

def load_colour_scheme(filename):
    with open(filename) as fp:
        data = fp.read().split('\n')[2:]


    class_names, class_ids, rs, gs, bs, = zip(*(vals.split()[:5] for vals in data[:-1]))
    class_ids = np.array(class_ids, dtype=np.uint16)
    number_of_classes = class_ids.max()+1
    retcolormap = VectorOfClassColour(int(number_of_classes))
    for name, cls_id, r, g, b in zip(class_names, class_ids, rs, gs, bs):
        retcolormap[int(cls_id)] = ClassColour(name, int(r, 10), int(g, 10), int(b, 10))

    return retcolormap

cnn_skip_frames = 10
use_crf = False
crf_skip_frames = 500
crf_iterations = 10


caffe = CaffeInterface()
caffe.Init("/mnt/1C562D12562CEDE8/MODELS/semanticFusion/nyu_rgbd/inference.prototxt", "/mnt/1C562D12562CEDE8/MODELS/semanticFusion/nyu_rgbd/inference.caffemodel", True)
num_classes = caffe.num_output_classes()
# num_classes = 14

semantic_fusion = SemanticFusionInterface(num_classes, 100)
class_colour_lookup = load_colour_scheme("/mnt/1C562D12562CEDE8/COMMONDEPS/semanticfusion2/class_colour_scheme.data")

width = 640
height = 480
Resolution.getInstance(width, height)
Intrinsics.getInstance(528, 528, 320, 240)
log_reader = PNGLogReader(
    "/mnt/1C562D12562CEDE8/DATASETS/nyu_data_small/bathroom_0003.txt",
    "/mnt/1C562D12562CEDE8/DATASETS/nyu_data_small/output_predictions.txt")
# log_reader = PNGLogReader(
#     "/mnt/1C562D12562CEDE8/DATASETS/nyu_data_small/output_predictions.txt",
#     "/mnt/1C562D12562CEDE8/DATASETS/nyu_data_small/bathroom_0003.txt",)

gui = Gui(True,class_colour_lookup,640,480)
_map = ElasticFusionInterface()
if not _map.Init(class_colour_lookup):
    print("Could not init map")

frame_num = 0

# log_reader.getNext()
# image_out = np.empty(log_reader.getImageSize(), dtype=np.uint8)
# log_reader.updateRGBVector(image_out)
# depth_it = np.empty(log_reader.getDepthSize()//2, dtype=np.uint16)
# log_reader.updateDepthVector(depth_it)
# reshaped_depth_it = depth_it.reshape((height,width))
# Image.fromarray(reshaped_depth_it.astype(np.uint8)).show()
# Image.fromarray((reshaped_depth_it/30).astype(np.uint8)).show()
img_it = np.empty((), dtype=np.uint8)
depth_it = np.empty((), dtype=np.uint16)
while not gui.ShouldQuit() and log_reader.hasMore():
    gui.preCall()

    if not gui.paused() or gui.step():
        log_reader.getNext()
        if log_reader.getDepthSize()//2 != depth_it.size:
            depth_it = np.empty(log_reader.getDepthSize()//2, dtype=np.uint16)
        if log_reader.getImageSize() != img_it.size:
            img_it = np.empty(log_reader.getImageSize(), dtype=np.uint8)
        log_reader.updateRGBVector(img_it)
        log_reader.updateDepthVector(depth_it)

        _map.setTrackingOnly(gui.tracking())

        if not _map.ProcessFrameNumpy(img_it, depth_it, log_reader.timestamp):
            raise Exception("Elastic fusion lost!")
        # if not gui.tracking():
        #     semantic_fusion.UpdateProbabilityTable(_map)

        # We do not need to perform a CNN update every frame, we perform it every 'cnn_skip_frames'
        if frame_num == 0 or (frame_num > 1 and ((frame_num + 1) % cnn_skip_frames == 0)):
            if log_reader.hasDepthFilled():
                log_reader.updateDepthVector(depth_it, use_filled=True)
                segmented_prob = caffe.ProcessFrameNumpy(img_it, depth_it, height, width)
            else:
                segmented_prob = caffe.ProcessFrameNumpy(img_it, depth_it, height, width)

            semantic_fusion.UpdateProbabilities(segmented_prob, _map)
        if use_crf and frame_num % crf_skip_frames == 0:
            print("Performing CRF Update...")
            semantic_fusion.CRFUpdate(_map,crf_iterations)

        frame_num += 1 # origional implementation has this on indent back but im not sure about that
        #
        # gui.renderMap(_map)
        # gui.displayRawNetworkPredictions("pred", segmented_prob.mutable_gpu_data())
        # semantic_fusion.CalculateProjectedProbabilityMap(_map)
        # gui.displayArgMaxClassColouring("segmentation",semantic_fusion.get_rendered_probability().mutable_gpu_data(),
        #                                  num_classes,semantic_fusion.get_class_max_gpu().gpu_data(),
        #                                  semantic_fusion.max_num_components(),_map.GetSurfelIdsGpu(), 0.0)
        # gui.displayImg("raw", _map.getRawImageTexture())

    gui.postCall()