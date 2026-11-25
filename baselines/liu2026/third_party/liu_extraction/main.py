import os
import argparse
import datetime
import torch
from icecream import ic
import numpy as np

from utils.helper import open_json, save_argparse_to_json
from models.modelwrapper import ModelWrapper
from models.train_new_models_tf import create_model
from model_extraction import main_
import utils.cfg as cfg
import tensorflow as tf


def convert_model_to_float32(model):
    # This function converts the model to quantized model with float32 precision
    # Transform the model configuration to float32 precision
    config = model.get_config()

    for layer in config["layers"]:
        if "config" in layer:
            layer["config"]["dtype"] = "float32"

    new_model = tf.keras.Model.from_config(config)

    for layer in new_model.layers:
        if layer.weights:
            original_layer = model.get_layer(layer.name)
            layer.set_weights(original_layer.get_weights())

    return new_model


if __name__ == "__main__":
    # add arguments to parser
    parser = argparse.ArgumentParser(description="run attack")
    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=["mnist", "cifar10", "carlini"],
        type=str,
        help="The dataset to use.",
    )
    parser.add_argument("--model", default="", type=str, help="The model to use.")
    parser.add_argument("--device", default="cpu", type=str, help="The device to use.")
    parser.add_argument("--load_json", default="", type=str, help="The config to load.")
    parser.add_argument(
        "--layerID",
        type=int,
        default=2,
        help="The ID of your target layer (as enumerated in model.layers).",
    )
    parser.add_argument(
        "--tgtNeurons",
        nargs="+",
        default=None,
        help="Specific target neuron IDs, e.g. '0 10 240'",
    )
    parser.add_argument(
        "--eps",
        type=int,
        default=8,
        help="(Optional) Precision parameter for lastLayer method (wiggle size is 10^-eps). Recommended 3 < eps < 8. (But honestly I have never touched this.)",
    )
    parser.add_argument(
        "--quantized",
        type=int,
        default=2,
        help="(Optional) Whether quantized or not. If float16 then specify this as 1 and if float32 then specify this as 2 if float64 then specify this as 0. Beware that with float64 this will run the precision improvement function which is really not great for mnist models. Preferably choose float32.",
    )
    parser.add_argument(
        "--signRecoveryMethod",
        type=str,
        default="neuronWiggle",
        choices=["carlini", "neuronWiggle", "cheat"],
        help="(Optional) Whether NeuronWiggle should be used or Carlini's method should be used.",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="(Optional) extraction seed."
    )
    parser.add_argument(
        "--Count", type=int, default=3000, help="Number of critical points."
    )
    parser.add_argument(
        "--cheat_noise",
        type=int,
        default=0,
        help="Whether to cheat the noise issue or not. (0 for False, 1 for True)",
    )
    parser.add_argument(
        "--cheat_sign",
        type=int,
        default=1,
        help="Whether to cheat the sign or not.",
    )
    parser.add_argument(
        "--output_index",
        default=0,
        type=int,
        help="only for cifar10 dataset. The output neuron to use.",
    )
    parser.add_argument(
        "--real_attack", default=0, type=int, help="only cheat on noise or not"
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="(Optional) debug mode prints all details.",
    )

    args = parser.parse_args()

    # Load the config file to overwrite the default values
    if len(args.load_json) > 0:
        args = open_json(args, args.load_json)

    model_arr = args.model.split("/")
    args.output_path = os.path.join(cfg.output_path, args.dataset, model_arr[-1])
    os.makedirs(args.output_path, exist_ok=True)
    save_argparse_to_json(args, args.output_path, "extraction_args.json")

    # Load the model
    if len(args.model) > 0:
        model_pth = os.path.join(cfg.ckp_path, args.model, "args.json")
        model_ckp_pth = os.path.join(cfg.ckp_path, args.model, "model.keras")

        model_parser = argparse.ArgumentParser(description="model parser")
        model_parser.add_argument("--load_json", default="", type=str, help="")
        model_parser.add_argument(
            "--layerID",
            type=int,
            default=2,
            help="The ID of your target layer (as enumerated in model.layers).",
        )
        model_parser.add_argument(
            "--tgtNeurons",
            nargs="+",
            default=None,
            help="Specific target neuron IDs, e.g. '0 10 240'",
        )
        model_parser.add_argument(
            "--eps",
            type=int,
            default=8,
            help="(Optional) Precision parameter for lastLayer method (wiggle size is 10^-eps). Recommended 3 < eps < 8. (But honestly I have never touched this.)",
        )
        model_parser.add_argument(
            "--quantized",
            type=int,
            default=0,
            help="(Optional) Whether quantized or not. If float16 then specify this as 1 and if float32 then specify this as 2 if float64 then specify this as 0. Beware that with float64 this will run the precision improvement function which is really not great for mnist models. Preferably choose float32.",
        )
        model_parser.add_argument(
            "--signRecoveryMethod",
            type=str,
            default="neuronWiggle",
            choices=["carlini", "neuronWiggle", "cheat"],
            help="(Optional) Whether NeuronWiggle should be used or Carlini's method should be used.",
        )
        model_parser.add_argument(
            "--seed", type=int, default=-1, help="(Optional) extraction seed."
        )
        model_parser.add_argument(
            "--Count", type=int, default=3000, help="Number of critical points."
        )
        model_parser.add_argument(
            "--cheat_noise",
            type=int,
            default=0,
            help="Whether to cheat the noise issue or not. (0 for False, 1 for True)",
        )
        model_parser.add_argument(
            "--cheat_sign",
            type=int,
            default=0,
            help="Whether to cheat the sign or not.",
        )
        model_parser.add_argument(
            "--output_index", default=0, type=int, help="Which output neuron to use."
        )
        model_parser.add_argument(
            "--real_attack", default=0, type=int, help="only cheat on noise or not"
        )
        model_parser.add_argument(
            "--debug",
            type=int,
            default=0,
            help="(Optional) debug mode prints all details.",
        )

        model_args = model_parser.parse_args()
        model_args = open_json(model_args, model_pth)
        model_args.device = args.device
        save_argparse_to_json(model_args, args.output_path, "model_args.json")

        # Load the model
        ic(model_args)
        model = create_model(model_args)
        model = tf.keras.models.load_model(model_ckp_pth)

        print(model.summary())
        for l in model.layers:
            if len(l.get_weights()) > 0:
                w, b = l.get_weights()
                print("----------------------")
                print("Layer: ", l.name)
                print("The size of the weights: ", w.shape)
                print("Weights: ", w)
                print("The size of the biases: ", b.shape)
                print("Biases: ", b)
        print("Real Model has been loaded!")

        # Start the attack
        mmodel = model

        # truncate the model
        # if args.real_attack == 0 and args.quantized == 2:
        #     new_model = convert_model_to_float32(model)

        #     for w in new_model.weights:
        #         print(w.name, w.dtype)

        #     mmodel = new_model
        #     print("Model has been converted to float32 precision.")
        #     print(mmodel.summary())
        #     for l in mmodel.layers:
        #         if len(l.get_weights()) > 0:
        #             w, b = l.get_weights()
        #             print("----------------------")
        #             print("Layer: ", l.name)
        #             print("The size of the weights: ", w.shape)
        #             print("Weights: ", w)
        #             print("The size of the biases: ", b.shape)
        #             print("Biases: ", b)

        main_(args, model_args, mmodel)
