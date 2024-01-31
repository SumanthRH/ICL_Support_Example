import datasets 
from dataclasses import dataclass
from functools import partial
import os

SINGLE_COLUMN_DATASETS = {
    "cola",
    "comqa",
    "cq",
    "sst2",
}
seed = 20
MAX_EXAMPLES = 15000

@dataclass
class DatasetConfig:
    dataset_name: str
    num_columns: int = None
    prompt_prepend1: str = None
    column1: str = None
    answer_column: str = None
    column2: str = None
    prompt_prepend2: str = None

def load_up_args(dataset_name):
        # let's have a mapping for columns and prompt prepends for different nli datasets
    is_glue = True
    dataset_config = DatasetConfig(dataset_name)
    if dataset_name == "qnli":
        dataset_config.num_columns = 2
        dataset_config.prompt_prepend1 = "Question: "
        dataset_config.prompt_prepend2 = " Premise: "
        dataset_config.column1 = "question"
        dataset_config.column2 = "sentence"
        dataset_config.answer_column = "label"
    elif dataset_name == "sst2":
        dataset_config.num_columns = 1
        dataset_config.prompt_prepend1 = ""
        dataset_config.column1 = "sentence"
        dataset_config.answer_column = "label"
    elif dataset_name == "qqp":
        dataset_config.num_columns = 2
        dataset_config.prompt_prepend1 = "Query A:"
        dataset_config.column1 = "question1"
        dataset_config.column2 = "question2"
        dataset_config.prompt_prepend2 = " Query B:"
        dataset_config.answer_column = "label"
    elif dataset_name == "cola":
        dataset_config.num_columns = 1
        dataset_config.prompt_prepend1 = ""
        dataset_config.column1 = "sentence"
        dataset_config.answer_column = "label"
    elif dataset_name in ["rte", "mrpc", "wnli"]:
        dataset_config.num_columns = 2
        dataset_config.prompt_prepend1 = "Sentence A: "
        dataset_config.prompt_prepend2 = " Sentence B: "
        dataset_config.column1 = "sentence1"
        dataset_config.column2 = "sentence2"
        dataset_config.answer_column = "label"
    elif dataset_name in ["scitail", "snli", "mnli"]:
        dataset_config.num_columns = 2
        dataset_config.prompt_prepend1 = "Premise: "
        dataset_config.prompt_prepend2 = " Hypothesis: "
        dataset_config.column1 = "premise"
        dataset_config.column2 = "hypothesis"
        dataset_config.answer_column = "label"
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    if dataset_name in ["scitail", "snli"]:
        is_glue = False
    return dataset_config, is_glue

def make_prompt_for_hf(example, dataset_config, single_column):
    example["sentence"] = dataset_config.prompt_prepend1 + example[dataset_config.column1]
    if example["sentence"][-1] != ".":
        example["sentence"] += '.'
    if not single_column:
        example["sentence"] += dataset_config.prompt_prepend2 + example[dataset_config.column2] + "\n"
    example["sentence"] = example["sentence"].replace("\t", " ") # tab is our separator for csv
    return example



if __name__ == "__main__":
     
    dataset_name = "cola"
    train_save_dir = f"full_train_data/{dataset_name}"
    test_save_dir = f"test_data/{dataset_name}"
    for dirc in [train_save_dir, test_save_dir]:
        if not os.path.exists(dirc):
            os.mkdir(dirc)
    dataset_name = dataset_name
    single_column = dataset_name in SINGLE_COLUMN_DATASETS
    dataset_config, is_glue = load_up_args(dataset_name)
    if is_glue:
        if dataset_name == "mnli":
            dataset = datasets.load_dataset("glue", dataset_name, split="train")
        else:
            dataset = datasets.load_dataset("glue", dataset_name, split="train")
    else:
        if dataset_name == "scitail":
            dataset = datasets.load_dataset(dataset_name, "tsv_format", split="train")
        else:
            dataset = datasets.load_dataset(dataset_name, split="train")
    
    if len(dataset) > MAX_EXAMPLES:
        dataset = dataset.shuffle(seed=seed).select(range(MAX_EXAMPLES))

    
    mapper = partial(make_prompt_for_hf, dataset_config=dataset_config, single_column=single_column)
    dataset = dataset.map(mapper)

    dataset = dataset.train_test_split(test_size=500, seed=seed)

    train_save_path = f"{train_save_dir}/train.tsv"

    dataset["train"].select_columns(["sentence", "label"]).to_csv(train_save_path, sep="\t")


    save_path = f"{test_save_dir}/test.tsv"

    dataset["test"].select_columns(["sentence", "label"]).to_csv(save_path, sep="\t")

    
