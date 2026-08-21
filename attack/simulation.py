import time
import numpy as np

from .common import set_seed
from .extractors import decrypt_B2, decrypt_L1, decrypt_L_last, decrypt_W2
from .model_io import CPUOracleCounter, generate_target_model_cpu
from .verification import (
    unshuffle_and_verify_B2,
    unshuffle_and_verify_L1,
    unshuffle_and_verify_L_last,
    unshuffle_and_verify_W2,
)


def run_original_cpu_simulation(num_models=10, trials_per_model=5, model_shape=(64, 20, 20, 10), q=512, init_model_seed=0, init_extraction_seed=0):
    total_runs = num_models * trials_per_model
    print(f"\n{'='*75}")
    print(" Initiating original CPU-based iterative simulation pipeline")
    print(f"    - Target models: {num_models}")
    print(f"    - Trials per model: {trials_per_model}")
    print(f"    - Total simulations: {total_runs}")
    print(f"{'='*75}\n")

    metrics = {
        'L1_success': 0, 'W2_success': 0, 'B2_success': 0, 'L_last_success': 0,
        'L1_queries': [], 'L2_queries': [], 'L_last_queries': [], 'times': []
    }

    for model_idx in range(1, num_models + 1):
        print(f"\n\n{'*'*75}")
        print(f" Preparing target model and environment [{model_idx}/{num_models}]")
        print(f"{'*'*75}")

        model_filename = f"{model_shape}-{model_idx:02d}.npz"
        model_seed = init_model_seed + 100 * (model_idx - 1)
        W_true, B_true = generate_target_model_cpu(q, model_shape, save_path=model_filename, target_seed=model_seed)

        for trial in range(1, trials_per_model + 1):
            extraction_seed = init_extraction_seed + (trial - 1)
            set_seed(extraction_seed)
            print(f"\n>>> Starting extraction for Model {model_idx}, Trial {trial}/{trials_per_model} (Extraction seed: {extraction_seed}) <<<")

            oracle_obj = CPUOracleCounter(W_true, B_true, q, model_shape)
            start_time = time.time()

            try:
                dec_W1, dec_B1 = decrypt_L1(
                    oracle_obj.prob_oracle,
                    model_shape[0],
                    model_shape[1],
                    model_shape[-1],
                    q,
                    len(model_shape) - 1,
                    W1_anchors=1000,
                    W1_steps_per_anchor=10,
                    W1_queries=1,
                    B1_anchors=10,
                    B1_queries=500,
                )

                if dec_W1 is None:
                    print(f"Error: L1 extraction failed for Model {model_idx}, Trial {trial}.")
                    continue

                W1_recon, B1_recon, pi_1 = unshuffle_and_verify_L1(dec_W1, dec_B1, W_true[0], B_true[0], W_true[1], tolerance=1)

                if W1_recon is None:
                    print(f"Error: L1 verification failed for Model {model_idx}, Trial {trial}.")
                    continue

                metrics['L1_success'] += 1
                queries_used_until_L1 = oracle_obj.total_queries
                metrics['L1_queries'].append(queries_used_until_L1)

                dec_W2 = decrypt_W2(
                    oracle_obj.prob_oracle,
                    dec_W1,
                    model_shape[0],
                    model_shape[1],
                    model_shape[2],
                    model_shape[-1],
                    q,
                    len(model_shape) - 1,
                    max_anchors=1000,
                    steps_per_anchor=10,
                    p1_queries=1,
                    p2_queries=10,
                    p3_queries=1,
                )

                if dec_W2 is None:
                    print(f"Error: W2 extraction failed for Model {model_idx}, Trial {trial}.")
                    continue

                W2_recon, pi_2 = unshuffle_and_verify_W2(dec_W2, W_true[1], pi_1)

                if W2_recon is None:
                    print(f"Error: W2 verification failed for Model {model_idx}, Trial {trial}.")
                    continue

                metrics['W2_success'] += 1

                dec_B2 = decrypt_B2(
                    oracle_obj.prob_oracle,
                    dec_W1,
                    dec_B1,
                    dec_W2,
                    model_shape[0],
                    model_shape[1],
                    model_shape[2],
                    q,
                    num_bases=10,
                    test_queries=500,
                )

                B2_recon = unshuffle_and_verify_B2(dec_B2, B_true[1], pi_2, tolerance=1)

                if B2_recon is None:
                    print(f"Error: B2 verification failed for Model {model_idx}, Trial {trial}.")
                    continue

                metrics['B2_success'] += 1
                queries_used_until_L2 = oracle_obj.total_queries
                metrics['L2_queries'].append(queries_used_until_L2 - queries_used_until_L1)

                dec_W_last, dec_B_last = decrypt_L_last(oracle_obj.prob_oracle, [dec_W1, dec_W2], [dec_B1, dec_B2], model_shape[-1], q)

                is_match, W_last_recon, B_last_recon = unshuffle_and_verify_L_last(dec_W_last, dec_B_last, W_true[-1], B_true[-1], pi_2, tolerance=1)

                if not is_match:
                    print(f"Error: Final layer verification failed for Model {model_idx}, Trial {trial}.")
                    continue

                metrics['L_last_success'] += 1
                queries_used = oracle_obj.total_queries
                metrics['L_last_queries'].append(queries_used - queries_used_until_L2)
                elapsed = time.time() - start_time
                metrics['times'].append(elapsed)

                print(f"Extraction successful for Model {model_idx}, Trial {trial}. (Queries: {queries_used:,}, Time: {elapsed:.1f}s)")

            except Exception as e:
                print(f"Warning: Unexpected error during Model {model_idx}, Trial {trial}: {e}")
                continue

    print(f"\n{'='*55}")
    print(" Original Pipeline Simulation Summary")
    print(f"{'='*55}")
    print(f" Total target models: {num_models} (Initial model seed: {init_model_seed})")
    print(f" Total simulations: {total_runs} (Initial extraction seed: {init_extraction_seed})")
    print("-" * 55)
    print(f" L1 (W1, B1) Success Rate: {metrics['L1_success']:02d}/{total_runs} ({(metrics['L1_success']/total_runs)*100:.1f}%)")
    print(f" W2 Success Rate         : {metrics['W2_success']:02d}/{total_runs} ({(metrics['W2_success']/total_runs)*100:.1f}%)")
    print(f" B2 Success Rate         : {metrics['B2_success']:02d}/{total_runs} ({(metrics['B2_success']/total_runs)*100:.1f}%)")
    print(f" Final Layer Success Rate: {metrics['L_last_success']:02d}/{total_runs} ({(metrics['L_last_success']/total_runs)*100:.1f}%)")

    if metrics['L_last_queries']:
        print("-" * 55)
        print(f" Average queries (successful trials): {np.mean(metrics['L1_queries']):,.0f} + {np.mean(metrics['L2_queries']):,.0f} + {np.mean(metrics['L_last_queries']):,.0f}")
        print(f" Average time (successful trials): {np.mean(metrics['times']):.1f}s")
    print(f"{'='*55}")

    return metrics