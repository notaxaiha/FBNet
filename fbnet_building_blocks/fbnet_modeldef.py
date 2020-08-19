# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# please, end the file with '}' and nothing else. this file updated automatically

MODEL_ARCH = {
    # FBNet-A
    "fbnet_a": {
        "block_op_type": [
            ["skip"],                                               # stage 1
            ["ir_k3_e3"], ["ir_k3_e1"], ["skip"],     ["skip"],     # stage 2
            ["ir_k5_e6"], ["ir_k3_e3"], ["ir_k5_e1"], ["ir_k3_e3"], # stage 3
            ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k5_s2"], ["ir_k5_e6"], # stage 4
            ["ir_k3_e6"], ["ir_k5_s2"], ["ir_k5_e1"], ["ir_k3_s2"], # stage 5
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k5_e6"], # stage 6
            ["ir_k5_e6"],                                           # stage 7
        ],
        "block_cfg": {
            # [channel, stride] or [channel, stride, kernel]
            "first": [16, 2],
            "stages": [
                # [[expantion_ratio, channel, number_of_layers, stride_at_first_block_of_stage]]
                [[1, 16, 1, 1]],                                                        # stage 1
                [[3, 24, 1, 2]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  # stage 2
                [[6, 32, 1, 2]],  [[3, 32, 1, 1]],  [[1, 32, 1, 1]],  [[3, 32, 1, 1]],  # stage 3
                [[6, 64, 1, 2]],  [[3, 64, 1, 1]],  [[1, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                [[6, 112, 1, 1]], [[1, 112, 1, 1]], [[3, 112, 1, 1]], [[1, 112, 1, 1]], # stage 5
                [[6, 184, 1, 2]], [[6, 184, 1, 1]], [[3, 184, 1, 1]], [[6, 184, 1, 1]], # stage 6
                [[6, 352, 1, 1]],                                                       # stage 7
            ],
            "backbone": [num for num in range(23)],
        },
    },
    # FBNet-B
    "fbnet_b": {
        "block_op_type": [
            ["ir_k3_e1"],                                           # stage 1
            ["ir_k3_e6"], ["ir_k5_e1"], ["ir_k3_e1"], ["ir_k3_e1"], # stage 2
            ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k3_e6"], ["ir_k5_e6"], # stage 3
            ["ir_k5_e6"], ["ir_k5_e1"], ["skip"],     ["ir_k5_e3"], # stage 4
            ["ir_k5_e6"], ["ir_k3_e1"], ["ir_k5_e1"], ["ir_k5_e3"], # stage 5
            ["ir_k5_e6"], ["ir_k5_e1"], ["ir_k5_e6"], ["ir_k5_e6"], # stage 6
            ["ir_k3_e6"],                                           # stage 7
        ],
        "block_cfg": {
            # [channel, stride] or [channel, stride, kernel]
            "first": [16, 2],
            "stages": [
                # [[expantion_ratio, channel, number_of_layers, stride_at_first_block_of_stage]]
                [[1, 16, 1, 1]],                                                        # stage 1
                [[3, 24, 1, 2]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  # stage 2
                [[6, 32, 1, 2]],  [[3, 32, 1, 1]],  [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                [[6, 64, 1, 2]],  [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  [[3, 64, 1, 1]],  # stage 4
                [[6, 112, 1, 1]], [[1, 112, 1, 1]], [[3, 112, 1, 1]], [[3, 112, 1, 1]], # stage 5
                [[6, 184, 1, 2]], [[1, 184, 1, 1]], [[6, 184, 1, 1]], [[6, 184, 1, 1]], # stage 6
                [[6, 352, 1, 1]],                                                       # stage 7
            ],
            "backbone": [num for num in range(23)],
        },
    },
    # FBNet-C
    "fbnet_c": {
        "block_op_type": [
            ["ir_k3_e1"],                                           # stage 1
            ["ir_k3_e6"], ["skip"],     ["ir_k3_e1"], ["ir_k3_e1"], # stage 2
            ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k5_e6"], ["ir_k3_e6"], # stage 3
            ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k5_e6"], ["ir_k5_e6"], # stage 4
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e3"], # stage 5
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], # stage 6
            ["ir_k3_e6"],                                           # stage 7
        ],
        "block_cfg": {
            # [channel, stride] or [channel, stride, kernel]
            "first": [16, 2],
            "stages": [
                # [[expantion_ratio, channel, number_of_layers, stride_at_first_block_of_stage]]
                [[1, 16, 1, 1]],                                                        # stage 1
                [[3, 24, 1, 2]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  # stage 2
                [[6, 32, 1, 2]],  [[3, 32, 1, 1]],  [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                [[6, 64, 1, 2]],  [[3, 64, 1, 1]],  [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                [[6, 112, 1, 1]], [[6, 112, 1, 1]], [[6, 112, 1, 1]], [[3, 112, 1, 1]], # stage 5
                [[6, 184, 1, 2]], [[6, 184, 1, 1]], [[6, 184, 1, 1]], [[6, 184, 1, 1]], # stage 6
                [[6, 352, 1, 1]],                                                       # stage 7
            ],
            "backbone": [num for num in range(23)],
        },
    },
    # FBNet-96-0.35-1 - for input size 96 and channel scaling 0.35
    "fbnet_96_035_1": {
        "block_op_type": [
            ["ir_k3_e1"],                                   # stage 1
            ["ir_k3_e6"], ["ir_k3_e6"], ["skip"], ["skip"], # stage 2
            ["ir_k5_e6"], ["ir_k5_e6"], ["skip"], ["skip"], # stage 3
            ["ir_k5_e6"], ["skip"],     ["skip"], ["skip"], # stage 4
            ["ir_k3_e6"], ["skip"],     ["skip"], ["skip"], # stage 5
            ["ir_k5_e6"], ["ir_k5_e6"], ["skip"], ["skip"], # stage 6
            ["ir_k5_e6"],                                   # stage 7
        ],
        "block_cfg": {
            # [channel, stride] or [channel, stride, kernel]
            "first": [16, 2],
            "stages": [
                # [[expantion_ratio, channel, number_of_layers, stride_at_first_block_of_stage]]
                [[1, 16, 1, 1]],                                                        # stage 1
                [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  # stage 2
                [[6, 32, 1, 2]],  [[6, 32, 1, 1]],  [[1, 32, 1, 1]],  [[1, 32, 1, 1]],  # stage 3
                [[6, 64, 1, 2]],  [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                [[6, 112, 1, 1]], [[1, 112, 1, 1]], [[1, 112, 1, 1]], [[1, 112, 1, 1]], # stage 5
                [[6, 184, 1, 2]], [[6, 184, 1, 1]], [[1, 184, 1, 1]], [[1, 184, 1, 1]], # stage 6
                [[6, 352, 1, 1]],                                                       # stage 7
            ],
            "backbone": [num for num in range(23)],
        },
    },
    # FBNet-Samsung-S8
    "fbnet_samsung_s8": {
        "block_op_type": [
            ["ir_k3_e1"],                                           # stage 1
            ["ir_k3_e3"], ["ir_k3_e1"], ["skip"],     ["skip"],     # stage 2
            ["ir_k5_e6"], ["ir_k3_e3"], ["ir_k3_e3"], ["ir_k3_e6"], # stage 3
            ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k5_e3"], ["ir_k5_e3"], # stage 4
            ["ir_k3_e6"], ["ir_k5_e3"], ["ir_k5_e3"], ["ir_k5_e3"], # stage 5
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k5_e6"], # stage 6
            ["ir_k5_e6"],                                           # stage 7
        ],
        "block_cfg": {
            # [channel, stride] or [channel, stride, kernel]
            "first": [16, 2],
            "stages": [
                # [[expantion_ratio, channel, number_of_layers, stride_at_first_block_of_stage]]
                [[1, 16, 1, 1]],                                                        # stage 1
                [[3, 24, 1, 2]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  # stage 2
                [[6, 32, 1, 2]],  [[3, 32, 1, 1]],  [[3, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                [[6, 64, 1, 2]],  [[3, 64, 1, 1]],  [[3, 64, 1, 1]],  [[3, 64, 1, 1]],  # stage 4
                [[6, 112, 1, 1]], [[3, 112, 1, 1]], [[3, 112, 1, 1]], [[3, 112, 1, 1]], # stage 5
                [[6, 184, 1, 2]], [[6, 184, 1, 1]], [[3, 184, 1, 1]], [[6, 184, 1, 1]], # stage 6
                [[6, 352, 1, 1]],                                                       # stage 7
            ],
            "backbone": [num for num in range(23)],
        },
    },
    # FBNet-iPhoneX
    "fbnet_iphonex": {
        "block_op_type": [
            ["skip"],                                               # stage 1
            ["ir_k3_e6"], ["ir_k3_e1"], ["ir_k3_e1"], ["ir_k3_e1"], # stage 2
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e6"], # stage 3
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e3"], ["ir_k3_e6"], # stage 4
            ["ir_k3_e6"], ["ir_k3_e3"], ["ir_k3_e3"], ["ir_k3_e3"], # stage 5
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e3"], # stage 6
            ["ir_k5_e6"],                                           # stage 7
        ],
        "block_cfg": {
            # [channel, stride] or [channel, stride, kernel]
            "first": [16, 2],
            "stages": [
                # [[expantion_ratio, channel, number_of_layers, stride_at_first_block_of_stage]]
                [[1, 16, 1, 1]],                                                        # stage 1
                [[3, 24, 1, 2]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  # stage 2
                [[6, 32, 1, 2]],  [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                [[6, 64, 1, 2]],  [[6, 64, 1, 1]],  [[3, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                [[6, 112, 1, 1]], [[3, 112, 1, 1]], [[3, 112, 1, 1]], [[3, 112, 1, 1]], # stage 5
                [[6, 184, 1, 2]], [[6, 184, 1, 1]], [[6, 184, 1, 1]], [[3, 184, 1, 1]], # stage 6
                [[6, 352, 1, 1]],                                                       # stage 7
            ],
            "backbone": [num for num in range(23)],
        },
    },
    # Searched Architecture
    "fbnet_cpu_sample1": {
        "block_op_type": [
            ["ir_k5_e6"],
            ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"],
            ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e3"], ["skip"],
            ["ir_k5_e6"], ["ir_k3_e6"], ["skip"], ["skip"],
            ["ir_k5_e6"], ["ir_k5_e6"], ["skip"], ["skip"],
            ["ir_k5_e6"], ["skip"], ["skip"], ["skip"],
            ["skip"],
        ],
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                [[6, 16, 1, 1]],                                                        # stage 1
                [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                [[6, 32, 1, 2]],  [[6, 32, 1, 1]],  [[3, 32, 1, 1]],  [[1, 32, 1, 1]],  # stage 3
                [[6, 64, 1, 2]],  [[6, 64, 1, 1]],  [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                [[6, 112, 1, 1]], [[6, 112, 1, 1]], [[1, 112, 1, 1]], [[1, 112, 1, 1]], # stage 5
                [[6, 184, 1, 2]], [[1, 184, 1, 1]], [[1, 184, 1, 1]], [[1, 184, 1, 1]], # stage 6
                [[1, 352, 1, 1]],                                                       # stage 7
            ],
            "backbone": [num for num in range(23)],
        },
    },
    "fbnet_cpu_sample2": {
            "block_op_type": [
            ["ir_k5_e6"], 
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["ir_k5_e6"], ["skip"], ["skip"], 
            ["ir_k5_e6"], ["skip"], ["skip"], ["skip"], 
            ["ir_k5_e6"], ["skip"], ["skip"], ["skip"], 
            ["skip"], 
            ],
            "block_cfg": {
                "first": [16, 2],
                "stages": [
                    [[6, 16, 1, 1]],                                                            # stage 1
                    [[6, 24, 1, 2]],  [[6, 24, 1, 1]],      [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                    [[6, 32, 1, 2]],  [[6, 32, 1, 1]],      [[1, 32, 1, 1]],  [[1, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[6, 64, 1, 1]],      [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                    [[6, 112, 1, 1]], [[1, 112, 1, 1]],     [[1, 112, 1, 1]], [[1, 112, 1, 1]], # stage 5
                    [[6, 184, 1, 2]], [[1, 184, 1, 1]],     [[1, 184, 1, 1]], [[1, 184, 1, 1]], # stage 6
                    [[1, 352, 1, 1]],                                                           # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        },
    "mb2_example": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k3_e3"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 2],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[3, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "mb2": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_s2"], ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_e3"], ["ir_k5_e3"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k5_e6"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[1, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[3, 96, 1, 1]],         [[3, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[6, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test1": {
            "block_op_type": [
            ["ir_k3_e3"], 
            ["ir_k3_e1"], ["ir_k5_e6"], 
            ["skip"], ["ir_k5_e6"], ["ir_k3_e3"], 
            ["ir_k5_e3"], ["ir_k5_e6"], ["ir_k5_s2"], ["ir_k3_e3"], 
            ["ir_k3_s2"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_s2"], ["ir_k5_s2"], 
            ["ir_k5_s2"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[3, 16, 1, 1]],                                                        # stage 1
                        [[1, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[1, 32, 1, 2]],  [[6, 32, 1, 1]],          [[3, 32, 1, 1]],  # stage 3
                        [[3, 64, 1, 2]],  [[6, 64, 1, 1]],          [[1, 64, 1, 1]],  [[3, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test2": {
            "block_op_type": [
            ["ir_k3_e3"], 
            ["ir_k3_e1"], ["ir_k5_e6"], 
            ["skip"], ["ir_k5_e6"], ["ir_k3_e3"], 
            ["ir_k5_e3"], ["ir_k5_e6"], ["ir_k5_s2"], ["ir_k3_e3"], 
            ["ir_k3_s2"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_s2"], ["ir_k5_s2"], 
            ["ir_k5_s2"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[3, 16, 1, 1]],                                                        # stage 1
                        [[1, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[1, 32, 1, 2]],  [[6, 32, 1, 1]],          [[3, 32, 1, 1]],  # stage 3
                        [[3, 64, 1, 2]],  [[6, 64, 1, 1]],          [[1, 64, 1, 1]],  [[3, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test3": {
            "block_op_type": [
            ["ir_k3_e3"], 
            ["ir_k3_e1"], ["ir_k5_e6"], 
            ["skip"], ["ir_k5_e6"], ["ir_k3_e3"], 
            ["ir_k5_e3"], ["ir_k5_e6"], ["ir_k5_s2"], ["ir_k3_e3"], 
            ["ir_k3_s2"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_s2"], ["ir_k5_s2"], 
            ["ir_k5_s2"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[3, 16, 1, 1]],                                                        # stage 1
                        [[1, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[1, 32, 1, 2]],  [[6, 32, 1, 1]],          [[3, 32, 1, 1]],  # stage 3
                        [[3, 64, 1, 2]],  [[6, 64, 1, 1]],          [[1, 64, 1, 1]],  [[3, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test4": {
            "block_op_type": [
            ["ir_k3_e3"], 
            ["ir_k3_e1"], ["ir_k5_e6"], 
            ["skip"], ["ir_k5_e6"], ["ir_k3_e3"], 
            ["ir_k5_e3"], ["ir_k5_e6"], ["ir_k5_s2"], ["ir_k3_e3"], 
            ["ir_k3_s2"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_s2"], ["ir_k5_s2"], 
            ["ir_k5_s2"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[3, 16, 1, 1]],                                                        # stage 1
                        [[1, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[1, 32, 1, 2]],  [[6, 32, 1, 1]],          [[3, 32, 1, 1]],  # stage 3
                        [[3, 64, 1, 2]],  [[6, 64, 1, 1]],          [[1, 64, 1, 1]],  [[3, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test5": {
            "block_op_type": [
            ["ir_k3_e3"], 
            ["ir_k3_e1"], ["ir_k5_e6"], 
            ["skip"], ["ir_k5_e6"], ["ir_k3_e3"], 
            ["ir_k5_e3"], ["ir_k5_e6"], ["ir_k5_s2"], ["ir_k3_e3"], 
            ["ir_k3_s2"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_s2"], ["ir_k5_s2"], 
            ["ir_k5_s2"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[3, 16, 1, 1]],                                                        # stage 1
                        [[1, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[1, 32, 1, 2]],  [[6, 32, 1, 1]],          [[3, 32, 1, 1]],  # stage 3
                        [[3, 64, 1, 2]],  [[6, 64, 1, 1]],          [[1, 64, 1, 1]],  [[3, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0526_mb2": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e3"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k3_s2"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[6, 96, 1, 1]],         [[3, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "m2_orig": {
            "block_op_type": [
            ["ir_k3_e6"],
            ["ir_k3_e6"], ["ir_k3_e6"],
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e6"],
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e6"],
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e6"],
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e6"],
            ["ir_k3_e6"],
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[6, 96, 1, 1]],         [[3, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "FBNet_DoReFa_w2a2": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k3_e6"], ["ir_k5_e3"], ["ir_k5_e1"], 
            ["ir_k5_e6"], ["ir_k3_s2"], ["ir_k5_e3"], ["skip"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k3_e1"], 
            ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k5_e3"], 
            ["ir_k5_e6"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[3, 32, 1, 1]],          [[1, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[1, 64, 1, 1]],          [[3, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[3, 160, 1, 1]],         [[3, 160, 1, 1]], # stage 6
                        [[6, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test_200616": {
            "block_op_type": [
            ["skip"], 
            ["ir_k5_s2"], ["skip"], 
            ["ir_k5_e6"], ["ir_k3_e1"], ["ir_k3_e3"], 
            ["ir_k5_e6"], ["skip"], ["ir_k3_e1"], ["ir_k3_e1"], 
            ["ir_k5_e3"], ["ir_k5_e3"], ["skip"], 
            ["ir_k3_e1"], ["ir_k5_e1"], ["skip"], 
            ["ir_k3_s2"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[1, 16, 1, 1]],                                                        # stage 1
                        [[1, 24, 1, 2]],  [[1, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[1, 32, 1, 1]],          [[3, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[1, 64, 1, 1]],          [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                        [[3, 96, 1, 1]], [[3, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[1, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "testo": {
            "block_op_type": [
            ["ir_k3_e1"], 
            ["ir_k3_e1"], ["ir_k3_e3"], 
            ["ir_k5_s2"], ["ir_k3_e3"], ["skip"], 
            ["ir_k5_e6"], ["ir_k3_e3"], ["skip"], ["ir_k5_e1"], 
            ["ir_k5_e6"], ["skip"], ["skip"], 
            ["ir_k3_e3"], ["ir_k3_s2"], ["skip"], 
            ["ir_k5_s2"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[1, 16, 1, 1]],                                                        # stage 1
                        [[1, 24, 1, 2]],  [[3, 24, 1, 1]],  # stage 2
                        [[1, 32, 1, 2]],  [[3, 32, 1, 1]],          [[1, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[3, 64, 1, 1]],          [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[3, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "cifar10_ngumbel_600_schedule5_1_flop": {
            "block_op_type": [
            ["ir_k5_e1"], 
            ["ir_k3_e6"], ["ir_k3_e1"], 
            ["ir_k5_e3"], ["ir_k5_e6"], ["ir_k3_s2"], 
            ["ir_k5_e6"], ["skip"], ["ir_k3_e6"], ["ir_k3_e1"], 
            ["skip"], ["ir_k5_e3"], ["ir_k5_s2"], 
            ["skip"], ["skip"], ["skip"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[1, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[1, 24, 1, 1]],  # stage 2
                        [[3, 32, 1, 2]],  [[6, 32, 1, 1]],          [[1, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[1, 64, 1, 1]],          [[6, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[3, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[1, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "cifar10_ngumbel_600_schedule5_1": {
            "block_op_type": [
            ["ir_k5_e1"], 
            ["ir_k3_e6"], ["ir_k3_e1"], 
            ["ir_k5_e3"], ["ir_k5_e6"], ["ir_k3_s2"], 
            ["ir_k5_e6"], ["skip"], ["ir_k3_e6"], ["ir_k3_e1"], 
            ["skip"], ["ir_k5_e3"], ["ir_k5_s2"], 
            ["skip"], ["skip"], ["skip"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[1, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[1, 24, 1, 1]],  # stage 2
                        [[3, 32, 1, 2]],  [[6, 32, 1, 1]],          [[1, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[1, 64, 1, 1]],          [[6, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[3, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[1, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "cifar10_ngumbel_180_schedule5_1": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k3_e6"], 
            ["ir_k5_e3"], ["ir_k3_e3"], ["ir_k3_e3"], 
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e1"], 
            ["ir_k3_e6"], ["skip"], ["skip"], 
            ["skip"], ["skip"], ["skip"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[3, 32, 1, 2]],  [[3, 32, 1, 1]],          [[3, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[1, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "cifar100_ngumbel_180_schedule5_1_flop": {
            "block_op_type": [
            ["ir_k5_s2"], 
            ["ir_k3_s2"], ["ir_k5_e1"], 
            ["ir_k5_s2"], ["ir_k5_e6"], ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k3_e3"], ["ir_k5_s2"], ["ir_k3_e3"], 
            ["skip"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["skip"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[1, 16, 1, 1]],                                                        # stage 1
                        [[1, 24, 1, 2]],  [[1, 24, 1, 1]],  # stage 2
                        [[1, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[3, 64, 1, 1]],          [[1, 64, 1, 1]],  [[3, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[1, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "cifar100_ngumbel_180_schedule5_1": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k5_e1"], ["ir_k3_e6"], 
            ["ir_k5_s2"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e3"], ["ir_k5_e3"], 
            ["skip"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["skip"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[1, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[1, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[3, 64, 1, 1]],  [[3, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[1, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test": {
            "block_op_type": [
            ["ir_k5_e3"], 
            ["ir_k3_e6"], ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k3_e3"], ["ir_k3_e1"], 
            ["skip"], ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k3_e1"], 
            ["ir_k5_e6"], ["ir_k3_s2"], ["ir_k5_e6"], 
            ["ir_k5_e3"], ["ir_k3_e6"], ["ir_k5_s2"], 
            ["ir_k5_e6"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[3, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[3, 32, 1, 1]],          [[1, 32, 1, 1]],  # stage 3
                        [[1, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[3, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[6, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "fluctuation_exp_4": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k3_e6"], 
            ["ir_k5_e3"], ["ir_k3_e3"], ["ir_k3_e3"], 
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e1"], 
            ["ir_k3_e6"], ["skip"], ["skip"], 
            ["skip"], ["skip"], ["skip"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[3, 32, 1, 2]],  [[3, 32, 1, 1]],          [[3, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[1, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "ngumbel_test": {
            "block_op_type": [
            ["ir_k5_e3"], 
            ["ir_k3_e6"], ["ir_k3_e3"], 
            ["ir_k3_e6"], ["ir_k3_e3"], ["ir_k3_e3"], 
            ["ir_k3_e6"], ["ir_k3_e1"], ["skip"], ["skip"], 
            ["skip"], ["skip"], ["skip"], 
            ["skip"], ["ir_k5_s2"], ["skip"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[3, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[3, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[3, 32, 1, 1]],          [[3, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[1, 64, 1, 1]],          [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[1, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[1, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test_eval_min01": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e1"], 
            ["ir_k5_e6"], ["ir_k5_e1"], ["skip"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test_eval_min1": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_e3"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k3_e3"], ["ir_k3_e3"], 
            ["ir_k3_e6"], ["ir_k5_e1"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["skip"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[3, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[3, 64, 1, 1]],  [[3, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test_eval2": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k3_e6"], ["ir_k3_e1"], ["ir_k5_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], ["skip"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test_eval_300_600": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k3_s2"], 
            ["skip"], ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k3_e1"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test_eval_min01_300_600": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k3_e6"], 
            ["skip"], ["ir_k3_e6"], ["ir_k5_e1"], 
            ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e1"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test_eval_min01_10_600": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k5_e3"], 
            ["skip"], ["ir_k5_e3"], ["ir_k5_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e1"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[3, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[3, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test_config": {
            "block_op_type": [
            ["ir_k3_e3"], 
            ["ir_k3_e1"], ["skip"], 
            ["ir_k3_e3"], ["ir_k3_s2"], ["skip"], 
            ["ir_k3_e3"], ["ir_k3_e3"], ["skip"], ["skip"], 
            ["ir_k5_e6"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["ir_k3_s2"], ["skip"], 
            ["ir_k3_e1"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[3, 16, 1, 1]],                                                        # stage 1
                        [[1, 24, 1, 2]],  [[1, 24, 1, 1]],  # stage 2
                        [[3, 32, 1, 2]],  [[1, 32, 1, 1]],          [[1, 32, 1, 1]],  # stage 3
                        [[3, 64, 1, 2]],  [[3, 64, 1, 1]],          [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test_exp_min01_180": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e1"], 
            ["ir_k5_e6"], ["skip"], ["skip"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test_exp_min1_180": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k3_e6"], 
            ["ir_k3_s2"], ["ir_k3_e3"], ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e6"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["skip"], 
            ["ir_k3_e1"], ["ir_k3_s2"], ["skip"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[1, 32, 1, 2]],  [[3, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[1, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "test_exp_min01_600": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k5_e1"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k5_e6"], 
            ["skip"], ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_s2"], ["ir_k5_e1"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[1, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
        "0815_test_eval_min01_step_15": {
            "block_op_type": [['ir_k3_e1'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_s2'], ['ir_k5_e6'], ['ir_k5_e3'], ['ir_k5_e6'], ['skip'], ['ir_k5_e6'], ['ir_k3_s2'], ['ir_k3_e3'], ['ir_k3_e3'], ['skip'], ['ir_k3_s2'], ['ir_k5_e1'], ['ir_k5_s2'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0815_test_eval_min01_step_40": {
            "block_op_type": [['ir_k3_e1'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e3'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e1'], ['skip'], ['ir_k3_e6'], ['ir_k3_e1'], ['skip'], ['ir_k3_s2'], ['skip'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0815_test_eval_min01_step_70": {
            "block_op_type": [['ir_k3_e1'], ['ir_k3_e6'], ['ir_k3_s2'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['skip'], ['ir_k3_e6'], ['ir_k3_e6'], ['skip'], ['ir_k3_e6'], ['skip'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },

    "0815_test_eval_min01_step_95": {
            "block_op_type": [['ir_k3_e1'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e3'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['skip'], ['ir_k3_e6'], ['skip'], ['skip'], ['skip'], ['ir_k3_e1'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0815_test_eval_min01_step_125": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['skip'], ['skip'], ['ir_k3_e6'], ['ir_k5_s2'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0815_test_eval_min01_step_150": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e3'], ['ir_k5_e6'], ['ir_k5_e1'], ['ir_k5_e1'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0815_test_eval_min01_step_180": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e3'], ['ir_k5_e6'], ['ir_k5_e1'], ['ir_k5_e1'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0815_test_eval2_step_15": {
            "block_op_type": [['ir_k3_e1'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e1'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['skip'], ['ir_k5_e6'], ['ir_k3_s2'], ['ir_k3_e3'], ['ir_k3_e3'], ['skip'], ['ir_k3_s2'], ['ir_k5_e1'], ['ir_k5_s2'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0815_test_eval2_step_40": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e3'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e1'], ['skip'], ['ir_k3_e6'], ['ir_k5_e1'], ['ir_k5_s2'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0815_test_eval2_step_70": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0815_test_eval2_step_95": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0815_test_eval2_step_125": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['skip'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_s2'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0815_test_eval2_step_150": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e1'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0815_test_eval2_step_180": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e3'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min01_cifar100": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k3_e6"], ["ir_k5_e3"], ["ir_k5_e3"], 
            ["ir_k3_e6"], ["ir_k5_e1"], ["skip"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[3, 96, 1, 1]],         [[3, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min1_cifar100": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k5_e1"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["skip"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[1, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[1, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval2_cifar100": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k3_e6"], ["ir_k5_e3"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k5_e1"], 
            ["skip"], 
                ],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[6, 96, 1, 1]], [[3, 96, 1, 1]],         [[6, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[3, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
        "0817_test_eval2_cifar100_15": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e1'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e1'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e1'], ['skip'], ['ir_k3_e1'], ['ir_k5_e3'], ['skip'], ['skip'], ['ir_k3_e6'], ['skip'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval2_cifar100_40": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e3'], ['skip'], ['ir_k5_e6'], ['ir_k5_e1'], ['ir_k5_e1'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval2_cifar100_70": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e3'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e3'], ['ir_k5_e1'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval2_cifar100_95": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e3'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e3'], ['ir_k5_e1'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval2_cifar100_125": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e3'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e3'], ['ir_k5_e1'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval2_cifar100_150": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e3'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e3'], ['ir_k5_e1'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval2_cifar100_180": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e3'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e3'], ['ir_k5_e1'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min1_cifar100_15": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e1'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e1'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k5_e3'], ['ir_k3_e1'], ['skip'], ['skip'], ['skip'], ['ir_k3_e6'], ['skip'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min1_cifar100_40": {
            "block_op_type": [['ir_k3_e6'], ['ir_k5_e1'], ['ir_k3_e3'], ['ir_k5_e6'], ['ir_k3_e3'], ['ir_k5_e3'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_s2'], ['ir_k3_e3'], ['ir_k3_e6'], ['skip'], ['skip'], ['ir_k3_e6'], ['skip'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min1_cifar100_70": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_s2'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e3'], ['ir_k5_e3'], ['ir_k3_e6'], ['skip'], ['skip'], ['ir_k3_e6'], ['skip'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min1_cifar100_95": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['skip'], ['skip'], ['ir_k3_e6'], ['skip'], ['skip'], ['ir_k3_s2']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min1_cifar100_125": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e3'], ['ir_k3_e6'], ['ir_k3_e6'], ['skip'], ['skip'], ['ir_k3_e6'], ['skip'], ['skip'], ['ir_k5_s2']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min1_cifar100_150": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e1'], ['skip'], ['ir_k3_e6'], ['skip'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min1_cifar100_180": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e1'], ['skip'], ['ir_k3_e6'], ['skip'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min01_cifar100_15": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e1'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e1'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e1'], ['ir_k5_e3'], ['ir_k3_e1'], ['ir_k5_e3'], ['ir_k3_e6'], ['skip'], ['ir_k3_e6'], ['skip'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min01_cifar100_40": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e3'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['skip'], ['skip'], ['ir_k3_e6'], ['skip'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min01_cifar100_70": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e3'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e1'], ['ir_k3_e6'], ['ir_k5_e1'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min01_cifar100_95": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e1'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min01_cifar100_125": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e3'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e1'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min01_cifar100_150": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e3'], ['ir_k5_e3'], ['ir_k3_e6'], ['ir_k5_e1'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    "0817_test_eval_min01_cifar100_180": {
            "block_op_type": [['ir_k3_e6'], ['ir_k3_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e6'], ['ir_k5_e6'], ['ir_k3_e6'], ['ir_k5_e3'], ['ir_k5_e3'], ['ir_k3_e6'], ['ir_k5_e1'], ['skip'], ['skip']],
                "block_cfg": {
                    "first": [32, 1],
                    "stages": [
                        [[6, 16, 1, 1]],                                                        # stage 1
                        [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  # stage 2
                        [[6, 32, 1, 2]],  [[6, 32, 1, 1]],          [[6, 32, 1, 1]],  # stage 3
                        [[6, 64, 1, 2]],  [[6, 64, 1, 1]],          [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                        [[1, 96, 1, 1]], [[6, 96, 1, 1]],         [[1, 96, 1, 1]], # stage 5
                        [[6, 160, 1, 2]], [[6, 160, 1, 1]],         [[1, 160, 1, 1]], # stage 6
                        [[1, 320, 1, 1]],                                                       # stage 7
                    ],
                    "backbone": [num for num in range(18)],
                },
            },
    }    
