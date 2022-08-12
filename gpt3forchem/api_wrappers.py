# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/01_api_wrappers.ipynb.

# %% auto 0
__all__ = ['fine_tune', 'query_gpt3', 'extract_prediction']

# %% ../notebooks/01_api_wrappers.ipynb 4
def fine_tune(train_file, valid_file, model: str = "ada"):
    # run the fine tuning
    subprocess.run(
        f"openai api fine_tunes.create -t {train_file} -v {valid_file} -m {model}",
        shell=True,
    )
    # sync runs with wandb
    subprocess.run("openai wandb sync -n 1", shell=True)


# %% ../notebooks/01_api_wrappers.ipynb 6
def query_gpt3(model, df, temperature=0, max_tokens=10):
    completions = []
    for i, row in df.iterrows():
        completion = openai.Completion.create(
            model=model,
            prompt=row["prompt"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        completions.append(completion)
        time.sleep(5)

    return completions

# %% ../notebooks/01_api_wrappers.ipynb 7
def extract_prediction(completion):
    return completion["choices"][0]["text"].split("@")[0].strip()

