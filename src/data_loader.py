from datasets import load_dataset

def load_mrpc():
    return load_dataset("glue", "mrpc")

def load_sts():
    return load_dataset("glue", "stsb")


def load_qqp():
    return load_dataset("glue", "qqp")

def load_paws():
    return load_dataset("paws", "labeled_final")
