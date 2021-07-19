import datasets
import torch
from transformers import BertModel, BertTokenizerFast, EncoderDecoderModel
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import pandas as pd

from data_preparation import dataset_filepath


def main():
    alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
    alephbert = BertModel.from_pretrained('onlplab/alephbert-base')
    alephbert.eval()

    x = alephbert_tokenizer("שלום נועה מור", return_tensors='pt')
    print(x)
    out = alephbert(**x)
    print(out)

    # with torch.no_grad():
    #     inp = alephbert_tokenizer("בדיקה", return_tensors="pt")
    #     out = alephbert(**inp)
    #     print(out)


def main_tokenizer_test():
    alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
    print(alephbert_tokenizer)


def main_model_test():
    alephbert = BertModel.from_pretrained('onlplab/alephbert-base')
    alephbert.eval()

    print(alephbert)


def main_seq2seq():
    alephbert_model_name = 'onlplab/alephbert-base'
    tokenizer = BertTokenizerFast.from_pretrained(alephbert_model_name)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(alephbert_model_name, alephbert_model_name,
                                                                tie_encoder_decoder=True)

    # forward
    input_ids = tokenizer("בדיקה", return_tensors="pt").input_ids
    outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)
    print(outputs, '\n')

    # training
    outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
    loss, logits = outputs.loss, outputs.logits
    print(loss, '\n', logits)

    # generation
    generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)
    print(tokenizer.decode(generated[0]))


def main_seq2seq_simple():
    alephbert_model_name = 'onlplab/alephbert-base'
    tokenizer = BertTokenizerFast.from_pretrained(alephbert_model_name)
    model = Seq2SeqModel('bert', encoder_name=alephbert_model_name, decoder_name=alephbert_model_name,
                         use_cuda=False)

    train_data = pd.read_csv('data/dummy/train.txt')\
        .applymap(lambda cell_text: tokenizer(cell_text, return_tensors='pt').input_ids)
    model.train(train_data, output_dir='out_dir')
    print(model)


if __name__ == '__main__':
    # main()
    # main_tokenizer_test()
    # main_model_test()
    # main_seq2seq()
    # main_seq2seq_simple()
    df = pd.read_csv('data/dummy/train.txt')
    print(df.head())
