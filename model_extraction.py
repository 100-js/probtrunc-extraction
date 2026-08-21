from attack import *


if __name__ == "__main__":
    run_original_cpu_simulation(
        num_models=5,
        trials_per_model=10,
        model_shape=(64, 20, 20, 10),
        q=512,
        init_model_seed=0,
        init_extraction_seed=0,
    )