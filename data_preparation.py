import pandas as pd
from transformers import BertTokenizerFast

data_path = 'data/dummy'


def dataset_filepath(dataset: str, tokenized=False):
    suffix = '.txt'
    if tokenized:
        suffix = '_tokenized' + suffix
    return f'data/dummy/{dataset}{suffix}'


def text_to_tokens(dataset):
    tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
    dataset_filename = dataset_filepath(dataset)
    df = pd.read_csv(dataset_filename) \
        .applymap(lambda sentence: tokenizer(sentence).input_ids)

    tokenized_filename = dataset_filepath(dataset, tokenized=True)
    df.to_csv(tokenized_filename, index=False)


def main():
    # dataset = 'train'
    dataset = 'validation'
    # dataset = 'test'

    text_to_tokens(dataset)


if __name__ == '__main__':
    main()
