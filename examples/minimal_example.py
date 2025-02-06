from hposuite.study import create_study

if __name__ == "__main__":
    study = create_study(
        name="minimal_example",
        output_dir="example-outputs",
    )
    study.optimize(
        optimizers=("DEHB", {"eta": 3}),
        benchmarks="mfh3_good",
        seeds=1,
        budget=50,
        overwrite=True,
    )