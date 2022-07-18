import pandas as pd

ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE = "what is the {property} of {text}###"
ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE = " {value}@@@"


def create_prompts(df, targets):
    prompts = []
    for _, row in df.iterrows():
        for target in targets:
            value = row[target]
            name = row["name"]
            prompts.append(
                {
                    "prompt": ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE.format(
                        property=target, text=name
                    ),
                    "completion": ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE.format(value=value),
                }
            )
    return pd.DataFrame(prompts)
