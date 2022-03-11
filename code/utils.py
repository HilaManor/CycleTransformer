import os
import re
import torch
import random
import numpy as np

def _gen_unique_out_dir_path(args):
    """
    Generates unique name for a new file, to avoid override.

    :param args: The current configuration of the test.

    :return: string of the possible path, with important data from args.
    """

    base_out = args['output_dir']
    possible_name = f"{args['db_type']}_e{args['epochs']}"
    possible_path = os.path.join(base_out, possible_name)

    if os.path.exists(possible_path):
        # rename with "name_name(num)"
        dirs = [f for f in os.listdir(base_out) if os.path.isdir(os.path.join(base_out, f))]

        ptrn = possible_name.replace('[', '\[').replace(']', '\]')
        matches = re.findall(ptrn + r'(\((\d+)\))?', '\n'.join(dirs))
        int_matches = [int(j) for i, j in matches if j]
        if int_matches:
            possible_name += f'({max(int_matches) + 1})'
        else:
            possible_name += '(1)'

        possible_path = os.path.join(base_out, possible_name)
    return possible_path


def create_output_dir(args):
    outpath = _gen_unique_out_dir_path(args)
    args['output_dir'] = outpath
    os.makedirs(outpath)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
