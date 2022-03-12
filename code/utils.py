"""General utility functions

function _gen_unique_out_dir_path - generate a unique name for each new out dir
function create_output_dir - create a new output dir
function set_seed - set a seed for reproducibility
function sentence_similarity - calculate similarity between sentences
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~
import os
import re
import torch
import random
import numpy as np
from sentence_transformers import util

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~~


def _gen_unique_out_dir_path(args):
    """Generates unique name for a new file, to avoid override.

    :param args: a dictionary containing configuration parameters for the entire model.
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
    """Create a new output dir with unique name

    :param args: a dictionary containing configuration parameters for the entire model.
    """
    out_path = _gen_unique_out_dir_path(args)
    args['output_dir'] = out_path
    os.makedirs(out_path)


def set_seed(seed=42):
    """Sets a seed for reproducibility using python, numpy, huggingface and pytorch methods

    :param seed: the seed number
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sentence_similarity(model, gen_sentences, gt_sentences):
    """Calculate the cosine similarity between sentences

    :param model: sentence transformer model
    :param gen_sentences: generated sentences
    :param gt_sentences: ground truth sentences
    :return: the cosine similarity between the generated sentences and the gt sentences a
    """
    gt_sentences_embeddings = model.encode(gt_sentences)
    gen_sentences_embeddings = model.encode(gen_sentences)

    return util.cos_sim(gt_sentences_embeddings, gen_sentences_embeddings)