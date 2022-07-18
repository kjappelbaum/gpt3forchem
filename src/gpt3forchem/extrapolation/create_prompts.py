import pandas as pd
import random

ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE = "what is the {property} of {text}###"
ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE = " {value}@@@"


def create_prompts(df, targets, zero_skip_probability=0.5):
    prompts = []
    for _, row in df.iterrows():
        for target in targets:
            value = row[target]
            if (value == 0) & (not "bandgap" in target):
                if random.random() > zero_skip_probability:
                    continue
            name = row["info.mofid.mofid"]
            target = target.replace("outputs.pbe.bandgap_binned", "bandgap")
            target = target.replace("_binned", "")

            prompts.append(
                {
                    "prompt": ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE.format(
                        property=target, text=name
                    ),
                    "completion": ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE.format(value=value),
                }
            )
    return pd.DataFrame(prompts)
