from hposuite.study import create_study

if __name__ == "__main__":
    study = create_study(
        name="minimal_example",
    )
    study.optimize(
        optimizers=("DEHB_Optimizer", {"eta": 3}),
        benchmarks="mfh3_good",
        seeds=1,
        budget=50,
        overwrite=True,
    )