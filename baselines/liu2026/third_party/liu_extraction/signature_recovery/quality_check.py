import sys
from io import StringIO
import numpy as np


# ----------
# Function for quality check. Called in neuronWiggle.py Signature Recovery part
# ----------
class DualStream:
    def __init__(self):
        self.console = sys.stdout
        self.stringio = StringIO()

    def write(self, message):
        self.console.write(message)
        self.stringio.write(message)

    def flush(self):
        self.console.flush()
        self.stringio.flush()

    def getvalue(self):
        return self.stringio.getvalue()


def check_quality(
    model,
    layer_num,
    extracted_normal,
    extracted_bias,
    dimOfLayer,
    critical_groups,
    do_fix=False,
):
    """
    Check the quality of the solution.

    The first function is read-only, and just reports how good or bad things are.
    The second half, when in cheating mode, will align the two matrices.
    """
    # Redirect stdout to capture print statements while printing to console
    dual_stream = DualStream()
    old_stdout = sys.stdout
    sys.stdout = dual_stream

    try:
        print("\nCheck the solution of the last weight matrix.")

        reorder = [None] * (dimOfLayer)
        for i in range(dimOfLayer):
            gaps = []
            ratios = []
            for j in range(dimOfLayer):
                if np.all(np.abs(extracted_normal[:, i])) < 1e-9:
                    extracted_normal[:, i] += 1e-9
                ratio = (
                    model.get_layer(index=layer_num + 1).get_weights()[0][:, j]
                    / extracted_normal[:, i]
                )
                ratio = np.median(ratio)
                error = (
                    model.get_layer(index=layer_num + 1).get_weights()[0][:, j]
                    - ratio * extracted_normal[:, i]
                )
                error = np.sum(error**2) / np.sum(
                    model.get_layer(index=layer_num + 1).get_weights()[0][:, j] ** 2
                )
                gaps.append(error)
                ratios.append(ratio)
            print(
                "Neuron",
                i,
                "maps on to neuron",
                np.argmin(gaps),
                "with error",
                np.min(gaps) ** 0.5,
                "ratio",
                ratios[np.argmin(gaps)],
            )
            print(
                "Bias check",
                (
                    model.get_layer(index=layer_num + 1).get_weights()[1][
                        np.argmin(gaps)
                    ]
                    - extracted_bias[i] * ratios[np.argmin(gaps)]
                ),
            )

            reorder[np.argmin(gaps)] = i
            if do_fix:
                extracted_normal[:, i] *= np.abs(ratios[np.argmin(gaps)])
                extracted_bias[i] *= np.abs(ratios[np.argmin(gaps)])

            if min(gaps) > 1e-2:
                print("ERROR LAYER EXTRACTED INCORRECTLY")
                print("\tGAPS:", " ".join("%.04f" % x for x in gaps))
                print(
                    "\t  Got:",
                    " ".join(
                        "%.04f" % x
                        for x in extracted_normal[:, i] / extracted_normal[0, i]
                    ),
                )
                print(
                    "\t Real:",
                    " ".join(
                        "%.04f" % x
                        for x in model.get_layer(index=layer_num + 1).get_weights()[0][
                            :, np.argmin(gaps)
                        ]
                        / model.get_layer(index=layer_num + 1).get_weights()[0][
                            0, np.argmin(gaps)
                        ]
                    ),
                )

        # Randomly assign the unused neurons.
        used = [x for x in reorder if x is not None]
        missed = list(set(range(len(reorder))) - set(used))
        for i in range(len(reorder)):
            if reorder[i] is None:
                reorder[i] = missed.pop()
        if do_fix and len(critical_groups) != 0:
            extracted_normal = extracted_normal[:, reorder]
            critical_groups = [critical_groups[i] for i in reorder]
            extracted_bias = extracted_bias[reorder]
        print("Real: ", model.get_layer(index=layer_num + 1).get_weights())
        print("Corrected: ", extracted_normal, extracted_bias)

        for i in range(dimOfLayer):
            # Compare the real weights to the extracted weights
            print("Original Neuron", i, "maps on to recovered neuron", reorder[i])
            real_weights = model.get_layer(index=layer_num + 1).get_weights()[0]
            unified_real_weights = real_weights[:, i] / real_weights[0, i]
            unified_read_bias = (
                model.get_layer(index=layer_num + 1).get_weights()[1][i]
                / real_weights[0, i]
            )
            # print(unified_real_weights)
            # print("Extracted")
            unified_extracted_weights = extracted_normal[:, i] / extracted_normal[0, i]
            unified_extracted_bias = extracted_bias[i] / extracted_normal[0, i]
            # print(unified_extracted_weights)
            distance1 = np.linalg.norm(unified_real_weights - unified_extracted_weights)
            distance2 = np.abs(unified_read_bias - unified_extracted_bias)
            if distance1 > 1e-2 or distance2 > 1e-2:
                print("ERROR LAYER EXTRACTED INCORRECTLY")
            else:
                print("absolute values of weights are close enough")
    finally:
        # Reset stdout
        sys.stdout = old_stdout

    # Get the string from DualStream object
    output = dual_stream.getvalue()
    return extracted_normal, extracted_bias, critical_groups, output
