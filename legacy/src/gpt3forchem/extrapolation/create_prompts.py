import pandas as pd
import random

ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE = "what is the {property} of {text}###"
ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE = " {value}@@@"


def create_prompts(df, targets, zero_skip_probability=0.98, add_context_to_prompts=True):
    prompts = []
    for _, row in df.iterrows():
        for target in targets:
            value = row[target]
            if (value == 0) & (not "bandgap" in target):
                if random.random() < zero_skip_probability:
                    continue
            name = row["info.mofid.mofid"]
            target = target.replace("outputs.pbe.bandgap_binned", "bandgap")
            target = target.replace("_binned", "")

            if add_context_to_prompts:
                if "H2O" in target:
                    target = (
                        "Henry coefficient of H2O (water), a polar molecule with quadrupole moment"
                    )
                elif "H2S" in target:
                    target = "Henry coefficient of H2S (hydrogen disulfide), a polar molecule with quadrupole moment"
                elif "CO2" in target:
                    target = "Henry coefficient of CO2 (carbon dioxide), an unpolar molecule with quadrupole moment"
                elif "N2" in target:
                    target = "Henry coefficient of N2 (nitrogen), an unpolar molecule with quadrupole moment"
                elif "CH4" in target:
                    target = "Henry coefficient of CH4 (methane), an unpolar molecule without quadrupole moment"
                elif "Xe" in target:
                    target = "Henry coefficient of Xe (xenon), an unpolar noble gas"
                elif "Kr" in target:
                    target = "Henry coefficient of Kr (krypton), an unpolar noble gas"

            prompts.append(
                {
                    "prompt": ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE.format(
                        property=target, text=name
                    ),
                    "completion": ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE.format(value=value),
                }
            )
    return pd.DataFrame(prompts)
