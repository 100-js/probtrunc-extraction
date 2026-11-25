*This repository contains the code for the manuscript "Navigating the Deep: End-to-End Extraction on Deep Neural Networks".*

# Navigating the Deep: End-to-End Extraction on Deep Neural Networks
This repository contains an implementation of the first end-to-end, polynomial-time cryptanalytic model extraction on a ReLU-based deep neural network (DNN). It builds on the codebases of [Carlini et al.](https://github.com/google-research/cryptanalytic-model-extraction), [Canales-Martínez et al.](https://github.com/Crypto-TII/deti), and [Foerster et al.](https://github.com/hannafoe/cryptanalytical-extraction), reusing and adapting parts of their implementations. 

We then introduce our own algorithms to: (1) address two key limitations in the signature-extraction step of prior approaches: deep merges and rank deficiency; (2) further improve the precision of extracted weights, thereby enhancing extraction efficiency and scalability to large architectures; and (3) combine two polynomial sign-extraction methods, the system-of-equations (SOE) and neuron wiggle approaches, to avoid relying on low-confidence neurons.

The extraction now also runs all in jax. Tensorflow is only used as an initial carrier for the weights and bias. We are using Python version 3.10.15.

## Reproduce extractions from our manuscript
To reproduce the extraction results presented in Tables 1-5 of our manuscript, please take the following command as an example:
```
python main.py --load_json scripts/attack/mnist/MNIST_8x8_1_linear.json --seed 42 --Count 3000 --layerID 2 --cheat_sign 1 --real_attack 0
```
- `--load_json`: specifies the model configuration file. You may replace it with any of the following examples:
    - `scripts/attack/mnist/MNIST_8x8_1_linear.json`
    - `scripts/attack/mnist/MNIST_16x8_1_linear.json`
    - `scripts/attack/cifar10/CIFAR10_8x8_softmax.json`
- `--seed`: sets the random seed. Any integer can be used, but we use 42 in all experiments to ensure reproducibility.
- `--Count`: determines the number of critical points used for the attack. It is fixed to 3000 for all layers except the first hidden layer, where it is 500.
- `--layerID`: indicates the index of the target layer. The first hidden layer is indexed as 1, the second is 2, and so on. The output layer is the last layer.
- `--cheat_sign`: controls whether to recover the signs of the absolute weights. If set to 0, the signs are recovered using our approach that combines the SOE and neuron wiggle methods. It is always used in end-to-end attacks. If set to 1, the signs are cheated by querying the target model. This option is used in the case of extracting a single layer, to isolate the performance of our weight-extraction algorithms from the sign-extraction step.
- `--real_attack`: controls the attack mode. If set to 0, the attack assumes all previous layers are fully recovered. It focuses on extracting signatures from the target layer. If set to 1, the attack will be applied to the extracted previous layers. It runs as an end-to-end attack. If set to 3, the attack performs only sign recovery, applying our SOE+Wiggle method under either the confidence or extension strategy. If set to 4, the attack performs only sign recovery, using Neuron Wiggle alone.

The three models used in our manuscript are provided in the `models` folder. The models are trained on the MNIST and CIFAR10 datasets. Additionally, the large model (3072-256×8-10) used to validate our combined sign recovery approach is the same model employed by [Canales-Martínez et al.](https://github.com/Crypto-TII/deti) and also included in the `models` folder.

## Instructions for running the code
### 1. Install dependencies
To get started you will need to install some dependencies. It should suffice to run the following command:
```
pip install -r requirements.txt
```

### 2. Prepare the target models:
To generate a model for extraction, run the following command:
```
python models/train_new_models_tf.py --load_json <config_file> 
```
- <config_file> should be replaced with the actual configuration path. For example, `scripts/generate_model/mnist/dense/8x8_linear.json`.
- The generated model will be saved in the `~/workspace/results/model_extraction/checkpoints/models/<model_name>` folder.

### 3. Attack the model layer by layer
To extract the signature from a specific layer of the target model, run the following command:
```
python main.py --load_json <config_file> --seed <random_seed> --Count <critical_point_number> --layerID <target_layer_id> --cheat_sign <0/1> --real_attack <0/1>
```
- <config_file> should be replaced with the actual configuration path. For example, `scripts/attack/mnist/MNIST_8x8_1_linear.json`.
- `--seed` option allows to try extraction on different random seeds.
- `--Count` option specifies the number of critical points to be used for the extraction.
- `--layerID` option specifies the target layer id. 
- `--cheat_sign` option determines whether to recover the signs of the absolute weights by our approach or by querying the target model.
- `--real_attack` option determines whether to perform an actual end-to-end attack.
- The output will be saved in the `~/workspace/results/model_extraction/output/<model_name>/extract results/Seed<random_seed>-Count<critical_point_number>` folder.
- The extracted signatures will be saved in the `~/workspace/results/model_extraction/output/<model_name>/recovered results/Seed<random_seed>-Count<critical_point_number>-RealAttack<0/1>` folder.

### 4. Evaluate the attack
To reproduce the evaluation results presented in Tables 3 and 5 and Figure 8 of our manuscript, please refer to the following instructions.

#### Compute input, plus-plus, and plus-minus activations
To compute the input activation, plus-plus activation, and plus-minus activation of the target model, run the following command:
```
python visualization/unreachable_weights_test.py --load_json <config_file> --random_points_number <random_point_number>
```
- <config_file> should be replaced with the actual configuration path. For example, `scripts/attack/mnist/MNIST_8x8_1_linear.json`.
- `--random_points_number` option specifies the number of random points to be used for the evaluation. Typically, we use 200000 random points.
- The output will be saved in the `~/workspace/results/model_extraction/output/Cheating_test/results_random_points/models/<model_name>/<random_point_number>` folder.

#### Evaluate Layer Coverage and Model Coverage
To evaluate the layer coverage and model coverage, use the following command:
```
python visualization/compute_difference.py --seed <random_seed> --Count <critical_point_number> --delta <random_input_point_number> --load_json <config_file>
```
- `--seed` option specfies the seed for extaction. To keep the results consistent, we use a fixed seed 42 for all runs.
- `--Count` option specifies the number of critical points used for the extraction. With `--seed` option, we can locate the extracted models to evaluate the layer coverage and model coverage.
- `--delta` option specifies the number of random input points to be used for the evaluation. Typically, we use 10^8 random input points.
- <config_file> should be replaced with the actual configuration path. For example, `scripts/attack/mnist/MNIST_8x8_1_linear.json`.

# Related Work

We gratefully thank the authors from the previous works for publishing their code to make our work possible:

- [Carlini et al., Cryptanalytic extraction of Neural Networks Models, CRYPTO, 2020.](https://arxiv.org/abs/2003.04884), [Link to their code](https://github.com/google-research/cryptanalytic-model-extraction)
- [Canales-Martínez et al., Polynomial Time Cryptanalytic Extraction of Neural Network Models, EUROCRYPT, 2023.](https://eprint.iacr.org/2023/1526), [Link to their code](https://github.com/Crypto-TII/deti)
- [Foerster et al., Beyond Slow Signs in High-fidelity Model Extraction, NeurIPS, 2024.](https://arxiv.org/abs/2406.10011), [Link to their code](https://github.com/hannafoe/cryptanalytical-extraction)
