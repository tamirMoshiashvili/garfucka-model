import datasets
import pandas as pd


def mkqa_to_csv():
    mkqa = pd.DataFrame(datasets.load_dataset('mkqa'))

    df = pd.DataFrame()
    df['input_text'] = mkqa.applymap(lambda cell: cell['queries']['he'])
    df['target_text'] = mkqa.applymap(lambda cell: cell['answers']['he'][0]['text'])
    df = df[df.target_text.notnull()]

    print(df.head())
    df.to_csv('data/mkqa/he.txt', index=False)
    # pd.DataFrame(mkqa).to_csv('data/mkqa/full.txt')


def mkqa_read_csv():
    pass


if __name__ == '__main__':
    mkqa_to_csv()
    mkqa_read_csv()
