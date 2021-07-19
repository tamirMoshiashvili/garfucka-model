import datasets
import pandas as pd
from transformers import BertTokenizerFast, EncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import os


max_length = 16
alephbert_model_name = 'onlplab/alephbert-base'
tokenizer = BertTokenizerFast.from_pretrained(alephbert_model_name)
batch_size = 8
out_dir = 'model'
rouge = datasets.load_metric("rouge")


def dataset_filepath(dataset: str, tokenized=False):
    suffix = '.txt'
    if tokenized:
        suffix = '_tokenized' + suffix
    return f'data/dummy/{dataset}{suffix}'


def process_data_to_model_inputs(batch: pd.DataFrame):
    global tokenizer

    # tokenize the inputs and labels
    inputs = tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=max_length)
    outputs = tokenizer(batch["target_text"], padding="max_length", truncation=True, max_length=max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels]
                       for labels in batch["labels"]]

    return batch


def prepare_dataset(dataset):
    dataset = dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["input_text", "target_text"]
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )
    return dataset


def config_model(model):
    global tokenizer

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.encoder.vocab_size

    model.config.max_length = max_length
    model.config.min_length = 1
    model.config.no_repeat_ngram_size = 2
    model.config.early_stopping = True
    model.config.length_penalty = 4.0
    # model.config.num_beams = 4


def get_training_args():
    os.environ['WANDB_DISABLED'] = "true"

    return Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=False,
        output_dir="./",
        logging_steps=5,
        save_steps=8,
        eval_steps=8,
        num_train_epochs=8
        # logging_steps=1000,
        # save_steps=500,
        # eval_steps=7500,
        # warmup_steps=2000,
        # save_total_limit=3,
    )


def compute_metrics(pred):
    global tokenizer, rouge

    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


def get_trainer(model, train_data, training_args, val_data):
    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer
    )


def train():
    # load datasets
    train_data = datasets.load_dataset('csv', data_files='data/garfuck/train.txt', split='train')
    val_data = datasets.load_dataset('csv', data_files='data/garfuck/validation.txt', split='train')
    train_data, val_data = prepare_dataset(train_data), prepare_dataset(val_data)

    # model
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(alephbert_model_name, alephbert_model_name,
                                                                tie_encoder_decoder=True)
    # model = EncoderDecoderModel.from_pretrained('model/alephseq2seq')
    config_model(model)

    # instantiate trainer
    training_args = get_training_args()
    trainer = get_trainer(model, train_data, training_args, val_data)
    trainer.train()
    model.save_pretrained('model/alephseq2seq')


def run():
    model = EncoderDecoderModel.from_pretrained('model/alephseq2seq')
    # model = EncoderDecoderModel.from_pretrained('model/alephseq2seq')
    config_model(model)

    while True:
        text = input("> ")

        inputs = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        outputs = model.generate(input_ids, attention_mask=attention_mask)

        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(output_str)


if __name__ == '__main__':
    # train()
    run()
