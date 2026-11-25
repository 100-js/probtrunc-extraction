import os
import json
import copy
import argparse

def open_json(args, load_json):
    with open(load_json, "r") as f:
        json_data = json.load(f)
        args.__dict__.update(json_data)
    return args


def save_argparse_to_json(args, log_dir, name="args.json"):
    args_dict = copy.deepcopy(vars(args))
    with open(os.path.join(log_dir, name), "w") as fp:
        json.dump(args_dict, fp, indent=4)

def str2bool(v):
    """
    Converts string to bool type; enables command line arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
def model_encoding(args):
    input_size = ""
    output_size = 1
    append = ""
    dataset = ""
    if args.dataset == "mnist":
        input_size = str(28**2) + "_"

    if args.dataset == "mnist":
        input_size = str(32**2) + "_"

    if args.dataset == "carlini":
        append = "_carlini"
        input_size = str(args.input_size) + "_"

    if args.lastactivation == "linear":
        output_size = 1

    if args.lastactivation == "softmax":
        output_size = 10

    return dataset, input_size, output_size, append