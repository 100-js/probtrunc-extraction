# All methods in here are a mesh of Carlini et al.'s, Canales-Martinez et al.'s, and Hanna et al.'s code and my adaptations
# Their code can be found:
# Carlini: https://github.com/google-research/cryptanalytic-model-extraction
# Canales-Martinez: https://anonymous.4open.science/r/deti-C405
# Hanna: https://github.com/hannafoe/cryptanalytical-extraction
# Beware that while we use tensorflow models as containers of parameters we do not actually use tensorflow in the extraction
# We use jax for extraction
# ---------------------------------------------------
# Prevent file locking errors
# ---------------------------------------------------
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ---------------------------------------------------
# Imports
# ---------------------------------------------------
import time
import numpy as np
import warnings
import multiprocessing as mp
import signal
import tensorflow as tf
import random
import logging

from whitebox import getWhiteboxRealSigns, getWhiteboxSignatures

from signature_recovery.utils import (
    forward,
    getAllWeightsAndBiases,
    predict_manual_fast,
)

from signature_recovery.utils import compute_difference, get_prev_ratio
from global_vars import setup_environment, set_global_vars
from signature_recovery.signature_recovery import recoverCritPts_Signature

# ---------------------------------------------------
# Set up logging
# ---------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------
# Tensorflow settings
# ---------------------------------------------------
warnings.filterwarnings("ignore")
setup_environment()

# Don't necessarily need this code but comment back in if needed:
# Prevent Tensorflow from gobbling the whole GPU memory
# devices = tf.config.list_physical_devices('GPU')
# for device in devices:
#    tf.config.experimental.set_memory_growth(device, True)


def detect_dead_weight_activation(
    args,
    model,
    point,
    recovered_weights,
    recovered_biases,
):
    # This function is used to check whether some dead weights are activated by a point
    x = point.copy()

    # The first layer must be fully recovered
    x = np.matmul(x, recovered_weights[0]) + recovered_biases[0]
    x = x * (x > 0)

    for i in range(1, len(recovered_weights)):
        prev_active_index = np.where(x != 0)[0]
        recovered_weight_matrix = np.array(recovered_weights[i])

        x = np.matmul(x, recovered_weights[i]) + recovered_biases[i]
        x = x * (x > 0)
        post_active_index = np.where(x != 0)[0]

        for j in range(recovered_weight_matrix.shape[1]):
            weight_array = recovered_weight_matrix[:, j].flatten()
            if np.any(weight_array[prev_active_index] == 0) and j in post_active_index:
                return True

    return False


def solve_final_layer(args, model, A, B, inputShape, lastLayerShape):
    """
    Solve the signature and sign directly for the last layer through a system of linear equations.
    """
    weights, biases = getAllWeightsAndBiases(model)
    # Critical Point Search
    # If we have already saved queries then we use them, else we need to create some
    points = []

    for i in range(100):
        offset = np.random.normal(0, 1, size=inputShape).flatten()
        direction = np.random.normal(0, 1, size=inputShape).flatten()
        x = random.uniform(-1000, 1000)

        point = offset + direction * x

        if args.real_attack == 0:
            points.append(point)
        else:
            if detect_dead_weight_activation(args, model, point, A, B):
                print("This point activates dead weights, skipping.")
                continue
            else:
                points.append(point)

    points = np.array(points)
    outs = predict_manual_fast(points, weights, biases)

    inputs, outputs = np.array(points), np.array(outs)

    # Actual Signature Recovery
    hidden = forward(inputs, A, B, with_relu=True)
    hidden = np.concatenate([hidden, np.ones((hidden.shape[0], 1))], axis=1)
    solution = np.linalg.lstsq(hidden, outputs)
    vector = solution[0]

    print("the hidden matrix is", hidden)
    print("the output matrix is", outputs)
    U, S, Vt = np.linalg.svd(hidden)
    print("The shape of hidden is", hidden.shape)
    print("The singular values of hidden are", S)
    print("The rank of hidden is", np.linalg.matrix_rank(hidden, tol=1e-6))

    extracted_normal = vector[:-1]
    extracted_bias = vector[-1]
    extracted_sign = np.ones(lastLayerShape)
    extracted_sequence = [i for i in range(0, lastLayerShape)]

    return (
        extracted_normal,
        extracted_bias,
        extracted_sign,
        extracted_sequence,
    )


def main_(args, model_args, model):
    logger.info(
        """
    # ---------------------------------------------------
    # Starting extracting weights and biases.
    # ---------------------------------------------------   
    """
    )
    logger.info(f"Parsed arguments for signature and sign recovery: \n\t {args}.")

    # ---------------------------------------------------
    # Filenames
    # ---------------------------------------------------
    def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    # In case the sign recovery should be carlini's
    if args.signRecoveryMethod == "carlini":
        args.nExp = -2
        if args.layerID == len(model.layers) - 2:
            set_global_vars(queries_bool=True)
    if args.layerID == len(model.layers) - 1:
        set_global_vars(queries_bool=True)

    # Output files should be marked with the appropriate settings
    modelname = args.model.replace(".keras", "")
    modelname = modelname.replace("/", "_")
    quant_level = ""
    if args.quantized == 1:
        quant_level = "_float16"
    elif args.quantized == 2:
        quant_level = "_float32"

    if args.signRecoveryMethod == "carlini":
        sign_str = "-carlini"
    if args.signRecoveryMethod == "neuronWiggle":
        sign_str = "-neuronWiggle"
    if args.signRecoveryMethod == "cheat":
        sign_str = "cheat"

    # Set the directory where recovered models will be saved
    savePath = os.path.join(
        args.output_path,
        "extract results",
        f"Seed{args.seed}-Count{args.Count}",
    )
    os.makedirs(savePath, exist_ok=True)
    print(f"Signature and Sign recovery results will be saved to \n\t {savePath}")

    # ---------------------------------------------------
    # Recover signatures (model extraction)
    # ---------------------------------------------------
    logger.info("Recovering signatures...")
    print("----------------------------------")
    print("Recovering signatures...")
    print("----------------------------------")

    # Set seed !!!! Do not set it if you are not testing
    np.random.seed(args.seed)
    random.seed(args.seed)

    inputShape = model.input_shape[1:]
    hiddenLayerIDs = [i for i in np.arange(1, len(model.layers) - 1)]

    # Start to recover the signatures
    # weights[:-1] and biases[:-1] are the **actual** weights and biases of layers 0 to layerID-1
    # weights[-1] and biases[-1] are the **simulated recovered** weights and biases of layer[layerID]
    weights, biases = getWhiteboxSignatures(model, args.layerID)

    if args.layerID < len(hiddenLayerIDs) + 1:
        # If we are not at the last layer
        print("Recovering signatures for layer ", args.layerID)
        (
            extracted_normal,
            extracted_bias,
            extracted_sign,
            extracted_sequence,
        ) = recoverCritPts_Signature(
            args,
            model,
            weights,
            biases,
            args.layerID,
            args.quantized,
            dataset=args.dataset,
            savePath=savePath,
        )
    else:
        # If we are at the last layer, we solve the system of linear equations directly
        print("Solving the last layer directly...")

        if args.real_attack == 0:
            # Assume all previous layers are correctly recovered
            final_start = time.time()
            (
                extracted_normal,
                extracted_bias,
                extracted_sign,
                extracted_sequence,
            ) = solve_final_layer(
                args,
                model,
                weights[:-1],
                biases[:-1],
                inputShape,
                model.layers[-1].output_shape[-1],
            )
            final_end = time.time()
            print("Time taken to solve final layer: ", final_end - final_start)

        if args.real_attack == 1:
            # Do the real end-to-end attack

            # Load recovered weights and biases
            A, B = [], []
            dead_neurons = []
            extracted_sequence = []

            # load the extracted weights and biases
            for i in range(1, args.layerID):
                print("Get recovered weights and biases for layer", i)
                if i == 1:
                    recoveryPath = os.path.join(
                        args.output_path,
                        "recovered results",
                        f"Seed{args.seed}-Count500-RealAttack{args.real_attack}",
                        # f"Seed{args.seed}-Count1000-RealAttack{args.real_attack}",
                    )
                else:
                    recoveryPath = os.path.join(
                        args.output_path,
                        "recovered results",
                        f"Seed{args.seed}-Count{args.Count}-RealAttack{args.real_attack}",
                    )
                recovered_parameters = np.load(
                    os.path.join(recoveryPath, f"recovered_{i}.npz")
                )
                for key in recovered_parameters.files:
                    print(key, recovered_parameters[key].dtype)

                A.append(recovered_parameters["extracted_normal"])
                B.append(recovered_parameters["extracted_bias"])
                extracted_sequence.append(recovered_parameters["extracted_sequence"])
                print(
                    "The sequence of columns in A is",
                    recovered_parameters["extracted_sequence"],
                )

                dead_neuron = []
                for index, b in enumerate(recovered_parameters["extracted_bias"]):
                    if abs(b) == 0:
                        dead_neuron.append(index)
                dead_neurons.append(dead_neuron)
            print("The dead neurons from the previous layer are", dead_neurons)

            (
                extracted_normal,
                extracted_bias,
                extracted_sign,
                extracted_sequence,
            ) = solve_final_layer(
                args,
                model,
                A,
                B,
                inputShape,
                model.layers[-1].output_shape[-1],
            )

    print("[Final Result]extracted_normal: ", extracted_normal)
    print("[Final Result]extracted_bias: ", extracted_bias)
    print("[Final Result]extracted_sign: ", extracted_sign)
    print("[Final Result]extracted_sequence: ", extracted_sequence)

    print("[Data type Checkpoint] Type of extracted_normal", extracted_normal.dtype)
    print("[Data type Checkpoint] Type of extracted_bias", extracted_bias.dtype)

    # check the quality of the extracted normal
    print("-----------------------------------------------")
    print("[Quality Checkpoint] Checking the quality of the extracted normal...")
    components_imprecision = []
    for index in range(extracted_normal.shape[1]):
        print("-----------------------------------------------")
        print("Checking neuron (component)", index, "of the extracted normal")
        extracted_weight = extracted_normal[:, index].flatten()
        real_weight = np.array(weights[-1][:, extracted_sequence[index]]).flatten()

        previous_extracted_sequence, previous_alpha = get_prev_ratio(
            args, model, biases
        )

        real_weight = real_weight[previous_extracted_sequence]

        if args.layerID == len(hiddenLayerIDs) + 1:
            scaled_extracted_weight = extracted_weight * previous_alpha
            diff, _, factor, sequence = compute_difference(
                scaled_extracted_weight, real_weight
            )
        else:
            real_weight = real_weight / previous_alpha
            diff, _, factor, sequence = compute_difference(
                extracted_weight, real_weight
            )

        components_imprecision.append(diff)
        print("[Quality Checkpoint] The imprecision of the extracted normal is", diff)

    print("-----------------------------------------------")
    components_imprecision = [value for value in components_imprecision if value != 0]
    print(
        "[Precision Checkpoint] The largest imprecision of the recovered components is",
        np.max(components_imprecision),
        "the smallest imprecision is",
        np.min(components_imprecision),
        "and the average imprecision is",
        np.mean(components_imprecision),
    )

    # breakpoint()

    # Store the extracted parameters
    recoveryPath = os.path.join(
        args.output_path,
        "recovered results",
        f"Seed{args.seed}-Count{args.Count}-RealAttack{args.real_attack}",
    )
    os.makedirs(recoveryPath, exist_ok=True)
    print(f"Recovery results will be saved to \n\t {recoveryPath}")

    np.savez(
        os.path.join(recoveryPath, f"recovered_{args.layerID}.npz"),
        extracted_normal=extracted_normal,
        extracted_bias=extracted_bias,
        extracted_sign=extracted_sign,
        extracted_sequence=extracted_sequence,
    )

    return None
