import os
import logging
import torch
from fbnet_building_blocks.fbnet_modeldef import MODEL_ARCH

class AverageMeter(object):
    def __init__(self, name=''):
        self._name = name
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def __str__(self):
        return "%s: %.5f" % (self._name, self.avg)
  
    def get_avg(self):
        return self.avg
    
    def __repr__(self):
        return self.__str__()

def weights_init(m, deepth=0, max_depth=2):
    if deepth > max_depth:
        return
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, torch.nn.BatchNorm2d):
        return
    elif isinstance(m, torch.nn.ReLU):
        return
    elif isinstance(m, torch.nn.Module):
        deepth += 1
        for m_ in m.modules():
            weights_init(m_, deepth)
    else:
        raise ValueError("%s is unk" % m.__class__.__name__)

def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('fbnet')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

# Example:
# utils.save(model, os.path.join(args.save, 'weights.pt'))
def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def load(model, model_path):
    print("model_path : ", model_path)
    model.load_state_dict(torch.load(model_path))

def add_text_to_file(text, file_path):
    with open(file_path, 'a') as f:
        f.write(text)
    
def clear_files_in_the_list(list_of_paths):
    for file_name in list_of_paths:
        open(file_name, 'w').close()

def create_directories_from_list(list_of_directories):
    for directory in list_of_directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
       
def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res

def check_tensor_in_list(atensor, alist):
    if any([(atensor == t_).all() for t_ in alist if atensor.shape == t_.shape]):
        return True
    return False

# Create an ARCH=model specification (see fbnet_building_blocks/fbnet_modeldef.py)
# and add the ARCH into MODEL_ARCH of fbnet_building_blocks/fbnet_modeldef.py with name "my_unique_name_for_ARCH"
# now, you can run the training procedure with new "my_unique_name_for_ARCH" specification
# Arguments:
# ops_names = ['ir_k3_e1', 'ir_k3_s2', 'ir_k3_e3', 'ir_k3_e6', 'ir_k5_e1',
#              'ir_k5_s2', 'ir_k5_e3', ..., 'ir_k5_e6', 'skip'] - list of 22 searched operations' names
# Note: don't read the code to understand it, just call the function for the following arguments
#       and look into fbnet_building_blocks/fbnet_modeldef.py
# ops_names = ["ir_k3_e1", "ir_k3_e6", "ir_k5_e1", "ir_k3_e1", "ir_k3_e1",  "ir_k5_e6", "ir_k5_e3",
#              "ir_k3_e6", "ir_k5_e6", "ir_k5_e6", "ir_k5_e1", "skip", "ir_k5_e3", "ir_k5_e6", "ir_k3_e1",
#              "ir_k5_e1", "ir_k5_e3", "ir_k5_e6", "ir_k5_e1", "ir_k5_e6", "ir_k5_e6", "ir_k3_e6"]
# my_unique_name_for_ARCH = "my_unique_name_for_ARCH"
def writh_new_ARCH_to_fbnet_modeldef(ops_names, my_unique_name_for_ARCH, supernet_type="mobilenetv2"):
    print('-- ops_names:', ops_names)
    print('-- my_unique_name_for_ARCH:', my_unique_name_for_ARCH)
    # assert len(ops_names) == 22
    if my_unique_name_for_ARCH in MODEL_ARCH:
        print("The specification with the name", my_unique_name_for_ARCH, "already written \
              to the fbnet_building_blocks.fbnet_modeldef. Please, create a new name \
              or delete the specification from fbnet_building_blocks.fbnet_modeldef (by hand)")
        assert my_unique_name_for_ARCH not in MODEL_ARCH
    
    ### create text to insert
    if supernet_type=='simple':

        text_to_write = "    \"" + my_unique_name_for_ARCH + "\": {\n\
                \"block_op_type\": [\n"

        ops = ["[\"" + str(op) + "\"], " for op in ops_names]
        ops_lines = [ops[0], ops[1]]
        ops_lines = [''.join(line) for line in ops_lines]
        text_to_write += '            ' + '\n            '.join(ops_lines)

        e = [(op_name[-1] if op_name[-2] == 'e' else '1') for op_name in ops_names]

        text_to_write += "\n\
                    ],\n\
                    \"block_cfg\": {\n\
                        \"first\": [32, 1],\n\
                        \"stages\": [\n\
                            [[" + e[0] + ", 160, 1, 1]],       # stage 1\n\
                            [[" + e[1] + ", 320, 1, 1]],     # stage 2\n\
                        ],\n\
                        \"backbone\": [num for num in range(3)],\n\
                    },\n\
                },\n\
        }\
        "

    elif supernet_type=='resnet':
        text_to_write = "    \"" + my_unique_name_for_ARCH + "\": {\n\
                \"block_op_type\": [\n"

        ops = ["[\"" + str(op) + "\"], " for op in ops_names]
        ops_lines = [ops[0:2], ops[2:4], ops[4:6], ops[6:8]]
        ops_lines = [''.join(line) for line in ops_lines]
        text_to_write += '            ' + '\n            '.join(ops_lines)

        e = [(op_name[-1] if op_name[-2] == 'e' else '1') for op_name in ops_names]

        text_to_write += "\n\
                    ],\n\
                    \"block_cfg\": {\n\
                        \"first\": [64, 2],\n\
                        \"stages\": [\n\
                            [[0, 64, 1, 1]],        # stage 1\n\
                            [[0, 128, 1, 1]],       # stage 2\n\
                            [[0, 128, 1, 2]],       # stage 3\n\
                            [[0, 256, 1, 1]],       # stage 4\n\
                            [[0, 256, 1, 2]],       # stage 5\n\
                            [[0, 512, 1, 1]],       # stage 6\n\
                            [[0, 512, 1, 2]],       # stage 7\n\
                            [[0, 512, 1, 1]],       # stage 8\n\
                        ],\n\
                        \"backbone\": [num for num in range(9)],\n\
                    },\n\
                },\n\
        }\
        "
    
    else:
        text_to_write = "    \"" + my_unique_name_for_ARCH + "\": {\n\
                \"block_op_type\": [\n"

        ops = ["[\"" + str(op) + "\"], " for op in ops_names]
        ops_lines = [ops[0], ops[1:3], ops[3:6], ops[6:10], ops[10:13], ops[13:16], ops[16]]
        ops_lines = [''.join(line) for line in ops_lines]
        text_to_write += '            ' + '\n            '.join(ops_lines)

        e = [(op_name[-1] if op_name[-2] == 'e' else '1') for op_name in ops_names]

        text_to_write += "\n\
                    ],\n\
                    \"block_cfg\": {\n\
                        \"first\": [32, 1],\n\
                        \"stages\": [\n\
                            [[" + e[0] + ", 16, 1, 1]],                                                        # stage 1\n\
                            [[" + e[1] + ", 24, 1, 1]],  [[" + e[2] + ", 24, 1, 1]],  # stage 2\n\
                            [[" + e[3] + ", 32, 1, 2]],  [[" + e[4] + ", 32, 1, 1]],  \
            [[" + e[5] + ", 32, 1, 1]],  # stage 3\n\
                            [[" + e[6] + ", 64, 1, 2]],  [[" + e[7] + ", 64, 1, 1]],  \
            [[" + e[8] + ", 64, 1, 1]],  [[" + e[9] + ", 64, 1, 1]],  # stage 4\n\
                            [[" + e[10] + ", 96, 1, 1]], [[" + e[11] + ", 96, 1, 1]], \
            [[" + e[12] + ", 96, 1, 1]], # stage 5\n\
                            [[" + e[13] + ", 160, 1, 2]], [[" + e[14] + ", 160, 1, 1]], \
            [[" + e[15] + ", 160, 1, 1]], # stage 6\n\
                            [[" + e[16] + ", 320, 1, 1]],                                                       # stage 7\n\
                        ],\n\
                        \"backbone\": [num for num in range(18)],\n\
                    },\n\
                },\n\
        }\
        "

    ### open file and find place to insert
    with open('./fbnet_building_blocks/fbnet_modeldef.py') as f1:
        lines = f1.readlines()
    end_of_MODEL_ARCH_id = next(i for i in reversed(range(len(lines))) if lines[i].strip() == '}')
    text_to_write = lines[:end_of_MODEL_ARCH_id] + [text_to_write]
    with open('./fbnet_building_blocks/fbnet_modeldef.py', 'w') as f2:
        f2.writelines(text_to_write)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

# TODO - Flops formula check
def count_conv_flop(layer, x):


    out_h = int(x.size()[2] / layer.stride[0])
    out_w = int(x.size()[3] / layer.stride[1])

    # print(f"f_h : {out_h}, f_w : {out_w}, c_in : {layer.in_channels}, c_out : {layer.out_channels}, k_h : {layer.kernel_size[0]}, k_w : {layer.kernel_size[1]}, g : {layer.groups}")

    delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * \
                out_h * out_w / layer.groups
    return delta_ops
