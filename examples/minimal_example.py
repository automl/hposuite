from hposuite.study import create_study

if __name__ == "__main__":
    study = create_study(
        name="minimal_example",
        output_dir="./examples/example-outputs",
        optimizers=("DEHB", {"eta": 3}),
        benchmarks="mfh3_good",
        seeds=1,
        budget=50,
    )
    study.optimize(
        overwrite=True,
    )