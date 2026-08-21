import argparse

from emax.simulation import random_matrices_scenario, trained_models_scenario


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run EMAX simulation scenarios.")
    parser.add_argument(
        "scenario",
        choices=("random", "trained"),
        default="random",
        const="random",
        nargs="?",
        help="Which scenario to run (default: random).",
    )
    parser.add_argument("--num-models", type=int, default=500, help="Number of models to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--q", type=int, default=512, help="Quantization scale.")
    parser.add_argument("--x-anchor", type=float, default=1e13, help="Base input magnitude.")
    parser.add_argument("--delta-x", type=float, default=1e10, help="Input perturbation size.")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for trained models.")
    parser.add_argument("--plot", action="store_true", help="Plot generated .dat results after the run.")

    args = parser.parse_args(argv)

    run_random = args.scenario in ("random")
    run_trained = args.scenario in ("trained")

    if run_random:
        random_matrices_scenario(
            layer_dims=(64, 20, 20, 10),
            q=args.q,
            X_anchor=args.x_anchor,
            Delta_X=args.delta_x,
            num_models=args.num_models,
            seed=args.seed,
        )

    if run_trained:
        trained_models_scenario(
            layer_dims=(64, 20, 20, 10),
            q=args.q,
            X_anchor=args.x_anchor,
            Delta_X=args.delta_x,
            num_models=args.num_models,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    if args.plot:
        from emax.visualization import plot_all_dat_results

        plot_all_dat_results()


if __name__ == "__main__":
    main()
