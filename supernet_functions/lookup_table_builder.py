import timeit
import torch
from collections import OrderedDict
import gc
from fbnet_building_blocks.fbnet_builder import PRIMITIVES
from general_functions.utils import add_text_to_file, clear_files_in_the_list
from supernet_functions.config_for_supernet import CONFIG_SUPERNET

# the settings from the page 4 of https://arxiv.org/pdf/1812.03443.pdf
#### table 2

##########################################
# ResNet 
##########################################
#CANDIDATE_BLOCKS = ["r_k3"]
#CANDIDATE_BLOCKS = ["r_k3", "r_k5" ]
CANDIDATE_BLOCKS = ["r_k3", "r_k5", "r_k7"]

# 4 layers
SEARCH_SPACE = OrderedDict([
    ("input_shape", [(64, 16, 16),
                     (64, 16, 16),
                     (128, 8, 8),
                     (128, 8, 8),
                     (256, 4, 4),
                     (256, 4, 4),
                     (512, 2, 2),
                     (512, 2, 2)]),
    ("channel_size", [64,
                      128,
                      128,
                      256,
                      256,
                      512,
                      512,
                      512]),
    ("strides", [1,
                 1,
                 2,
                 1,
                 2,
                 1,
                 2,
                 1])
])

'''
# 3 layers
SEARCH_SPACE = OrderedDict([
    ("input_shape", [(64, 16, 16),
                     (64, 16, 16),
                     (128, 8, 8),
                     (128, 8, 8),
                     (256, 4, 4),
                     (256, 4, 4)]),
    ("channel_size", [64,
                      128,
                      128,
                      256,
                      256,
                      256]),
    ("strides", [1,
                 1,
                 2,
                 1,
                 2,
                 1])
])
'''

'''
# 2 layers
SEARCH_SPACE = OrderedDict([
    ("input_shape", [(64, 16, 16),
                     (64, 16, 16),
                     (128, 8, 8),
                     (128, 8, 8)]),
    ("channel_size", [64,
                      128,
                      128,
                      128]),
    ("strides", [1,
                 1,
                 2,
                 1])
])

'''


'''
##########################################
# SimpleNet
##########################################
CANDIDATE_BLOCKS = ["s_k3", "s_k5", "s_k7"]
SEARCH_SPACE = OrderedDict([
    ("input_shape", [(32, 32, 32),
                     (160, 32, 32)]),
    ("channel_size", [160,
                      320]),
    ("strides", [1,
                 1])
])
'''


'''
##########################################
# MobileNet 
##########################################
CANDIDATE_BLOCKS = ["ir_k3_e1", "ir_k3_s2", "ir_k3_e3",
                    "ir_k3_e6", "ir_k5_e1", "ir_k5_s2",
                    "ir_k5_e3", "ir_k5_e6", "skip"]
SEARCH_SPACE = OrderedDict([
    #### table 1. input shapes of 22 searched layers (considering with strides)
    # Note: the second and third dimentions are recommended (will not be used in training) and written just for debagging
    # Imagenet - original
    # ("input_shape", [(32, 112, 112),
    #                  (16, 112, 112), (24, 56, 56),
    #                  (24, 56, 56), (32, 28, 28), (32, 28, 28),
    #                  (32, 28, 28), (64, 14, 14), (64, 14, 14), (64, 14, 14),
    #                  (64, 14, 14), (96, 14, 14), (96, 14, 14),
    #                  (96, 14, 14), (160, 7, 7), (160, 7, 7),
    #                  (160, 7, 7)]),

    # cifar-10
    ("input_shape", [(32, 32, 32),
                     (16, 32, 32), (24, 32, 32),
                     (24, 32, 32),   (32, 16, 16),  (32, 28, 16),
                     (32, 16, 16),   (64, 8, 8),  (64, 8, 8),  (64, 8, 8),
                     (64, 8, 8),   (96, 8, 8), (96, 8, 8),
                     (96, 8, 8),  (160, 4, 4),   (160, 4, 4),
                     (160, 4, 4)]),
    # table 1. filter numbers over the 22 layers
    ("channel_size", [16,
                      24,  24,
                      32,  32,  32,
                      64,  64,  64,  64,
                      96,  96,  96,
                      160, 160, 160,
                      320]),
    # table 1. strides over the 22 layers
    # mobiletnet v2 - cifar 10
    ("strides", [1,
                 1, 1,
                 2, 1, 1,
                 2, 1, 1, 1,
                 1, 1, 1,
                 2, 1, 1,
                 1])

    # # mobilenet v2 -imagenet - orig
    # ("strides", [1,
    #              2, 1,
    #              2, 1, 1,
    #              2, 1, 1, 1,
    #              1, 1, 1,
    #              2, 1, 1,
    #              1])
])
'''

# **** to recalculate latency use command:
# l_table = LookUpTable(calulate_latency=True, path_to_file='lookup_table.txt', cnt_of_runs=50)
# results will be written to './supernet_functions/lookup_table.txt''
# **** to read latency from the another file use command:
# l_table = LookUpTable(calulate_latency=False, path_to_file='lookup_table.txt')

# TODO - flops change
class LookUpTable:
    def __init__(self, candidate_blocks=CANDIDATE_BLOCKS, search_space=SEARCH_SPACE,
                 calulate_latency=False):
        self.cnt_layers = len(search_space["input_shape"])
        # constructors for each operation
        self.lookup_table_operations = {op_name : PRIMITIVES[op_name] for op_name in candidate_blocks}
        # arguments for the ops constructors. one set of arguments for all 9 constructors at each layer
        # input_shapes just for convinience
        self.layers_parameters, self.layers_input_shapes = self._generate_layers_parameters(search_space)
        
        # lookup_table
        self.lookup_table_flops = None
        # if calulate_latency:
        #     self._create_from_operations(cnt_of_runs=CONFIG_SUPERNET['lookup_table']['number_of_runs'],
        #                                  write_to_file=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table'])
        # else:
        #     self._create_from_file(path_to_file=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table'])
        #
        self._create_from_file(path_to_file=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table'])

    def _generate_layers_parameters(self, search_space):
        # layers_parameters are : C_in, C_out, expansion, stride
        layers_parameters = [(search_space["input_shape"][layer_id][0],
                              search_space["channel_size"][layer_id],
                              # expansion (set to -999) embedded into operation and will not be considered
                              # (look fbnet_building_blocks/fbnet_builder.py - this is facebookresearch code
                              # and I don't want to modify it)
                              -999,
                              search_space["strides"][layer_id]
                             ) for layer_id in range(self.cnt_layers)]
        
        # layers_input_shapes are (C_in, input_w, input_h)
        layers_input_shapes = search_space["input_shape"]
        
        return layers_parameters, layers_input_shapes
    
    # CNT_OP_RUNS us number of times to check latency (we will take average)
    # def _create_from_operations(self, cnt_of_runs, write_to_file=None):
    #     self.lookup_table_latency = self._calculate_latency(self.lookup_table_operations,
    #                                                         self.layers_parameters,
    #                                                         self.layers_input_shapes,
    #                                                         cnt_of_runs)
    #     if write_to_file is not None:
    #         self.write_lookup_table_to_file(write_to_file)
    #
    # def _calculate_latency(self, operations, layers_parameters, layers_input_shapes, cnt_of_runs):
    #     LATENCY_BATCH_SIZE = 1
    #     latency_table_layer_by_ops = [{} for i in range(self.cnt_layers)]
    #
    #     for layer_id in range(self.cnt_layers):
    #         for op_name in operations:
    #             op = operations[op_name](*layers_parameters[layer_id])
    #             input_sample = torch.randn((LATENCY_BATCH_SIZE, *layers_input_shapes[layer_id]))
    #             globals()['op'], globals()['input_sample'] = op, input_sample
    #             total_time = timeit.timeit('output = op(input_sample)', setup="gc.enable()", \
    #                                        globals=globals(), number=cnt_of_runs)
    #             # measured in micro-second
    #             latency_table_layer_by_ops[layer_id][op_name] = total_time / cnt_of_runs / LATENCY_BATCH_SIZE * 1e6
    #
    #     return latency_table_layer_by_ops
    #
    def write_lookup_table_to_file(self, path_to_file, flops_list):
        # candidate blocks
        clear_files_in_the_list([path_to_file])
        ops = [op_name for op_name in self.lookup_table_operations]
        text = [op_name + " " for op_name in ops[:-1]]
        text.append(ops[-1] + "\n")


        # for layer_id in range(self.cnt_layers):
        #     for op_name in ops:
        #         text.append(str(self.lookup_table_latency[layer_id][op_name]))
        #         text.append(" ")
        #     text[-1] = "\n"
        # text = text[:-1]

        for i in range(len(flops_list)):
            for j in range(len(flops_list[i])):
                text.append(str(flops_list[i][j]))
                text.append(" ")
            text[-1] = "\n"
        text = ''.join(text)

        add_text_to_file(text, path_to_file)
    
    def _create_from_file(self, path_to_file):
        self.lookup_table_flops = self._read_lookup_table_from_file(path_to_file)
    
    def _read_lookup_table_from_file(self, path_to_file):
        flops = [line.strip('\n') for line in open(path_to_file)]
        ops_names = flops[0].split(" ")
        flops = [list(map(float, layer.split(" "))) for layer in flops[1:]]
        
        lookup_table_flops= [{op_name : flops[i][op_id]
                                      for op_id, op_name in enumerate(ops_names)
                                     } for i in range(self.cnt_layers)]
        return lookup_table_flops
