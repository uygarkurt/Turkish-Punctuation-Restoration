import ast
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Sequence, ClassLabel, Dataset
from torcheval.metrics.functional import multiclass_f1_score, multiclass_precision, multiclass_recall, multiclass_accuracy

def data_load(df_path, label2id, label_list):
    df = pd.read_csv(df_path)
    df.insert(0, "id", "nan")
    df = df.rename({"sentence": "tokens"}, axis=1)
    for idx, row in tqdm(df.iterrows()):
        new_sentences = ast.literal_eval(row["tokens"])
        new_tags = [label2id[tag] for tag in ast.literal_eval(row["tags"])]

        df.at[idx, "id"] = str(idx)
        df.at[idx, "tokens"] = new_sentences
        df.at[idx, "tags"] = new_tags

    tags_classlabel = Sequence(feature=ClassLabel(names=label_list, num_classes=len(label_list)))
    ds = Dataset.from_pandas(df)
    ds.features["tags"] = tags_classlabel

    return(ds)

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def get_test_results(model, tokenizer, label_list, tokenized_punc_ds, device):
    final_preds = []
    final_golds = []
    for sample_idx in tqdm(range(len(tokenized_punc_ds["test"]))):
        torch.cuda.empty_cache()
        sample_dic = {
            "input_ids": torch.as_tensor([tokenized_punc_ds["test"][sample_idx]["input_ids"]]).to(device),
            "token_type_ids": torch.as_tensor([tokenized_punc_ds["test"][sample_idx]["token_type_ids"]]).to(device),
            "attention_mask": torch.as_tensor([tokenized_punc_ds["test"][sample_idx]["attention_mask"]]).to(device)}

        with torch.no_grad():
            logits = model(**sample_dic).logits

        preds = torch.argmax(logits, dim=2)[0] # Take 0th index since there's no batch
        encoded_tokens = tokenizer.convert_ids_to_tokens(tokenized_punc_ds["test"]["input_ids"][sample_idx])
        golds = tokenized_punc_ds["test"][sample_idx]["labels"]

        sep_idx = encoded_tokens.index("[SEP]")
        preds = preds[1: sep_idx]
        encoded_tokens = encoded_tokens[1: sep_idx]
        golds = golds[1: sep_idx]

        trimmed_preds = []
        trimmed_encoded_tokens = []
        trimmed_golds = []
        for pred, token, gold in zip(preds, encoded_tokens, golds):
            if gold != -100:
                trimmed_preds.append(pred)
                trimmed_encoded_tokens.append(token)
                trimmed_golds.append(gold)

        final_preds.append(trimmed_preds)
        final_golds.append(trimmed_golds)

    final_preds_flat = torch.as_tensor([item for sub_l in final_preds for item in sub_l])
    final_golds_flat = torch.as_tensor([item for sub_l in final_golds for item in sub_l])

    precision = multiclass_precision(final_golds_flat, final_preds_flat, num_classes=len(label_list), average=None)
    recall = multiclass_recall(final_golds_flat, final_preds_flat, num_classes=len(label_list), average=None)
    f1 = multiclass_f1_score(final_golds_flat, final_preds_flat, num_classes=len(label_list), average=None)

    res_dic = {"PERIOD": [float(precision[1]), float(recall[1]), float(f1[1])],
               "COMMA": [float(precision[2]), float(recall[2]), float(f1[2])],
               "QUESTION_MARK": [float(precision[3]), float(recall[3]), float(f1[3])]}

    res_df = pd.DataFrame.from_dict(res_dic, orient="index", columns=["PRECISION", "RECALL", "F1"])
    return res_df

def prepare_compute_metrics(label_list):
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = [ 
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)]
        true_predictions = [ 
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)]

        true_labels_flat_torch = torch.tensor([item for sub_l in true_labels for item in sub_l])
        true_predictions_flat_torch = torch.tensor([item for sub_l in true_predictions for item in sub_l])

        precision = multiclass_precision(true_labels_flat_torch, true_predictions_flat_torch, num_classes=len(label_list), average="macro")
        recall = multiclass_recall(true_labels_flat_torch, true_predictions_flat_torch, num_classes=len(label_list), average="macro")
        f1 = multiclass_f1_score(true_labels_flat_torch, true_predictions_flat_torch, num_classes=len(label_list), average="macro")
        accuracy = multiclass_accuracy(true_labels_flat_torch, true_predictions_flat_torch, num_classes=len(label_list), average="macro")

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy}
    return compute_metrics
