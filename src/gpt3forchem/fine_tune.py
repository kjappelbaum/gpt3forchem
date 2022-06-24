import subprocess


def fine_tune(train_file, valid_file, model: str = "ada"):
    # run the fine tuning
    subprocess.run(
        f"openai api fine_tunes.create -t {train_file} -v {valid_file} -m {model}", shell=True
    )
    # sync runs with wandb
    subprocess.run("openai wandb sync -n 1", shell=True)
