# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections import namedtuple


# --------------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.
        "trainId",  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!
        "category",  # The name of the category that this label belongs to
        "categoryId",  # The ID of this category. Used to create ground truth images
        # on category level.
        "hasInstances",  # Whether this label distinguishes between single instances or not
        "ignoreInEval",  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
        "color",  # The color of this label
    ],
)


# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels_cityscapes = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    Label("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
    Label("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
    Label("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
    Label("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
    Label("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
    Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    Label("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
    Label("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
    Label("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
    Label("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
    Label("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
    Label("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
    Label("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
    Label("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
    Label("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
    Label("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
    Label("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
    Label("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
    Label("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
    Label("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
    Label("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
    Label("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
    Label("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
    Label("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
    Label("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
    Label("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
    Label("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
    Label("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
    Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    Label("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
]

labels_replica = [
    #       name id trainId category catId hasInstances ignoreInEval color
    Label("backpack", 1, 0, "void", 0, False, False, (128, 0, 0)),
    Label("base-cabinet", 2, 1, "void", 0, False, False, (0, 128, 0)),
    Label("basket", 3, 2, "void", 0, False, False, (128, 128, 0)),
    Label("bathtub", 4, 3, "void", 0, False, False, (0, 0, 128)),
    Label("beam", 5, 4, "void", 0, False, False, (128, 0, 128)),
    Label("beanbag", 6, 5, "void", 0, False, False, (0, 128, 128)),
    Label("bed", 7, 6, "void", 0, False, False, (128, 128, 128)),
    Label("bench", 8, 7, "void", 0, False, False, (64, 0, 0)),
    Label("bike", 9, 8, "void", 0, False, False, (192, 0, 0)),
    Label("bin", 10, 9, "void", 0, False, False, (64, 128, 0)),
    Label("blanket", 11, 10, "void", 0, False, False, (192, 128, 0)),
    Label("blinds", 12, 11, "void", 0, False, False, (64, 0, 128)),
    Label("book", 13, 12, "void", 0, False, False, (192, 0, 128)),
    Label("bottle", 14, 13, "void", 0, False, False, (64, 128, 128)),
    Label("box", 15, 14, "void", 0, False, False, (192, 128, 128)),
    Label("bowl", 16, 15, "void", 0, False, False, (0, 64, 0)),
    Label("camera", 17, 16, "void", 0, False, False, (128, 64, 0)),
    Label("cabinet", 18, 17, "void", 0, False, False, (0, 192, 0)),
    Label("candle", 19, 18, "void", 0, False, False, (128, 192, 0)),
    Label("chair", 20, 19, "void", 0, False, False, (0, 64, 128)),
    Label("chopping-board", 21, 20, "void", 0, False, False, (128, 64, 128)),
    Label("clock", 22, 21, "void", 0, False, False, (0, 192, 128)),
    Label("cloth", 23, 22, "void", 0, False, False, (128, 192, 128)),
    Label("clothing", 24, 23, "void", 0, False, False, (64, 64, 0)),
    Label("coaster", 25, 24, "void", 0, False, False, (192, 64, 0)),
    Label("comforter", 26, 25, "void", 0, False, False, (64, 192, 0)),
    Label("computer-keyboard", 27, 26, "void", 0, False, False, (192, 192, 0)),
    Label("cup", 28, 27, "void", 0, False, False, (64, 64, 128)),
    Label("cushion", 29, 28, "void", 0, False, False, (192, 64, 128)),
    Label("curtain", 30, 29, "void", 0, False, False, (64, 192, 128)),
    Label("ceiling", 31, 30, "void", 0, False, False, (192, 192, 128)),
    Label("cooktop", 32, 31, "void", 0, False, False, (0, 0, 64)),
    Label("countertop", 33, 32, "void", 0, False, False, (128, 0, 64)),
    Label("desk", 34, 33, "void", 0, False, False, (0, 128, 64)),
    Label("desk-organizer", 35, 34, "void", 0, False, False, (128, 128, 64)),
    Label("desktop-computer", 36, 35, "void", 0, False, False, (0, 0, 192)),
    Label("door", 37, 36, "void", 0, False, False, (128, 0, 192)),
    Label("exercise-ball", 38, 37, "void", 0, False, False, (0, 128, 192)),
    Label("faucet", 39, 38, "void", 0, False, False, (128, 128, 192)),
    Label("floor", 40, 39, "void", 0, False, False, (64, 0, 64)),
    Label("handbag", 41, 40, "void", 0, False, False, (192, 0, 64)),
    Label("hair-dryer", 42, 41, "void", 0, False, False, (64, 128, 64)),
    Label("handrail", 43, 42, "void", 0, False, False, (192, 128, 64)),
    Label("indoor-plant", 44, 43, "void", 0, False, False, (64, 0, 192)),
    Label("knife-block", 45, 44, "void", 0, False, False, (192, 0, 192)),
    Label("kitchen-utensil", 46, 45, "void", 0, False, False, (64, 128, 192)),
    Label("lamp", 47, 46, "void", 0, False, False, (192, 128, 192)),
    Label("laptop", 48, 47, "void", 0, False, False, (0, 64, 64)),
    Label("major-appliance", 49, 48, "void", 0, False, False, (128, 64, 64)),
    Label("mat", 50, 49, "void", 0, False, False, (0, 192, 64)),
    Label("microwave", 51, 50, "void", 0, False, False, (128, 192, 64)),
    Label("monitor", 52, 51, "void", 0, False, False, (0, 64, 192)),
    Label("mouse", 53, 52, "void", 0, False, False, (128, 64, 192)),
    Label("nightstand", 54, 53, "void", 0, False, False, (0, 192, 192)),
    Label("pan", 55, 54, "void", 0, False, False, (128, 192, 192)),
    Label("panel", 56, 55, "void", 0, False, False, (64, 64, 64)),
    Label("paper-towel", 57, 56, "void", 0, False, False, (192, 64, 64)),
    Label("phone", 58, 57, "void", 0, False, False, (64, 192, 64)),
    Label("picture", 59, 58, "void", 0, False, False, (192, 192, 64)),
    Label("pillar", 60, 59, "void", 0, False, False, (64, 64, 192)),
    Label("pillow", 61, 60, "void", 0, False, False, (192, 64, 192)),
    Label("pipe", 62, 61, "void", 0, False, False, (64, 192, 192)),
    Label("plant-stand", 63, 62, "void", 0, False, False, (192, 192, 192)),
    Label("plate", 64, 63, "void", 0, False, False, (32, 0, 0)),
    Label("pot", 65, 64, "void", 0, False, False, (160, 0, 0)),
    Label("rack", 66, 65, "void", 0, False, False, (32, 128, 0)),
    Label("refrigerator", 67, 66, "void", 0, False, False, (160, 128, 0)),
    Label("remote-control", 68, 67, "void", 0, False, False, (32, 0, 128)),
    Label("scarf", 69, 68, "void", 0, False, False, (160, 0, 128)),
    Label("sculpture", 70, 69, "void", 0, False, False, (32, 128, 128)),
    Label("shelf", 71, 70, "void", 0, False, False, (160, 128, 128)),
    Label("shoe", 72, 71, "void", 0, False, False, (96, 0, 0)),
    Label("shower-stall", 73, 72, "void", 0, False, False, (224, 0, 0)),
    Label("sink", 74, 73, "void", 0, False, False, (96, 128, 0)),
    Label("small-appliance", 75, 74, "void", 0, False, False, (224, 128, 0)),
    Label("sofa", 76, 75, "void", 0, False, False, (96, 0, 128)),
    Label("stair", 77, 76, "void", 0, False, False, (224, 0, 128)),
    Label("stool", 78, 77, "void", 0, False, False, (96, 128, 128)),
    Label("switch", 79, 78, "void", 0, False, False, (224, 128, 128)),
    Label("table", 80, 79, "void", 0, False, False, (32, 64, 0)),
    Label("table-runner", 81, 80, "void", 0, False, False, (160, 64, 0)),
    Label("tablet", 82, 81, "void", 0, False, False, (32, 192, 0)),
    Label("tissue-paper", 83, 82, "void", 0, False, False, (160, 192, 0)),
    Label("toilet", 84, 83, "void", 0, False, False, (32, 64, 128)),
    Label("toothbrush", 85, 84, "void", 0, False, False, (160, 64, 128)),
    Label("towel", 86, 85, "void", 0, False, False, (32, 192, 128)),
    Label("tv-screen", 87, 86, "void", 0, False, False, (160, 192, 128)),
    Label("tv-stand", 88, 87, "void", 0, False, False, (96, 64, 0)),
    Label("umbrella", 89, 88, "void", 0, False, False, (224, 64, 0)),
    Label("utensil-holder", 90, 89, "void", 0, False, False, (96, 192, 0)),
    Label("vase", 91, 90, "void", 0, False, False, (224, 192, 0)),
    Label("vent", 92, 91, "void", 0, False, False, (96, 64, 128)),
    Label("wall", 93, 92, "void", 0, False, False, (224, 64, 128)),
    Label("wall-cabinet", 94, 93, "void", 0, False, False, (96, 192, 128)),
    Label("wall-plug", 95, 94, "void", 0, False, False, (224, 192, 128)),
    Label("wardrobe", 96, 95, "void", 0, False, False, (32, 0, 64)),
    Label("window", 97, 96, "void", 0, False, False, (160, 0, 64)),
    Label("rug", 98, 97, "void", 0, False, False, (32, 128, 64)),
    Label("logo", 99, 98, "void", 0, False, False, (160, 128, 64)),
    Label("bag", 100, 99, "void", 0, False, False, (32, 0, 192)),
    Label("set-of-clothing", 101, 100, "void", 0, False, False, (0, 0, 0)),
]

# --------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
# --------------------------------------------------------------------------------

trainId2color_cityscapes = {label.trainId: label.color for label in reversed(labels_cityscapes)}
numcolor2trainId_cityscapes = {
    label.color[0] + label.color[1] * 256 + label.color[2] * 256**2: label.trainId
    for label in reversed(labels_cityscapes)
    if label.trainId != 255
}
trainId2name_cityscapes = {label.trainId: label.name for label in reversed(labels_cityscapes) if label.trainId != 255}
trainId2Id_cityscapes = {label.trainId: label.id for label in reversed(labels_cityscapes) if label.trainId != 255}

trainId2color_replica = {label.trainId: label.color for label in reversed(labels_replica)}
numcolor2trainId_replica = {
    label.color[0] + label.color[1] * 256 + label.color[2] * 256**2: label.trainId
    for label in reversed(labels_replica)
    if label.trainId != 255
}
trainId2name_replica = {label.trainId: label.name for label in reversed(labels_replica) if label.trainId != 255}
trainId2Id_replica = {label.trainId: label.id for label in reversed(labels_replica) if label.trainId != 255}
