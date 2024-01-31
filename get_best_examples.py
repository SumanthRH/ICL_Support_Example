"""For a trained model, get the best examples from the pkl file"""

import os
import pickle
import numpy as np
import datasets
import yaml
from copy import deepcopy
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import PreservedScalarString as pss
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dqss
import argparse

def yaml_dumper(data, filepath):
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    yaml_dict = deepcopy(data)
    yaml_dict["examples"] = []
    
    for example in data["examples"]:
        prompt, solution = example["question"], example["answer"]
        yaml_dict["examples"].append({"question": pss(prompt), "answer": dqss(solution)})

    with open(filepath, "w") as f:
        yaml.dump(yaml_dict, f)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--pkl_path", type=str)
parser.add_argument("--save_dir", type=str, default="icl_few_shot_examples")


if __name__ == "__main__":
    args = parser.parse_args()
    
    dataset_name=args.dataset_name
    path = args.pkl_path

    is_glue = True
    if dataset_name in ["scitail", "snli"]:
        is_glue = False
    if is_glue:
        if dataset_name == "mnli":
            dataset = datasets.load_dataset("glue", dataset_name, split="validation_matched")
        else:
            dataset = datasets.load_dataset("glue", dataset_name, split="validation")
    else:
        if dataset_name == "scitail":
            dataset = datasets.load_dataset(dataset_name, "tsv_format", split="validation")
        else:
            dataset = datasets.load_dataset(dataset_name, split="validation")

    label_names = dataset.features["label"].names

    filename = "example_idxs_list_test_result.pkl"

    filepath = os.path.join(path, filename)
    with open(filepath, "rb") as f:
        data= pickle.load(f)

    best_ind = np.argmax(data["acc"])
    candidates = data["candidate_examples_list"][best_ind]
    candidates = [{"question": s, "answer": label_names[int(l)]} for s,l in candidates]

    save_yaml = {"task_name": dataset_name, "examples": candidates}
    yaml_dumper(save_yaml, os.path.join(args.save_dir, f"{dataset_name}_examples.yaml"))