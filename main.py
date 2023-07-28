import os
import torch
import timeit
import random
import warnings
import transformers
import numpy as np
from datetime import datetime
from datasets import DatasetDict
from transformers import BertTokenizerFast, BertForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer, AdamW, get_linear_schedule_with_warmup, pipeline

from utils import data_load, tokenize_and_align_labels, get_test_results, prepare_compute_metrics

transformers.utils.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"

gpu_count = torch.cuda.device_count()
print(f"Using: {device}")
print(f"Number of GPUs: {gpu_count}", "\n")
for i in range(gpu_count):
    print(torch.cuda.get_device_name(i))

id2label = { 
            0: "O",
            1: "PERIOD",
            2: "COMMA",
            3: "QUESTION_MARK",
}
label2id = {v: k for k, v in id2label.items()}
label_list = list(id2label.values())

tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-cased")
model = BertForTokenClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels = len(label_list), id2label = id2label, label2id = label2id)

train_path = "./multitarget-ted/en-tr/ted_train_tr.csv"
val_path = "./multitarget-ted/en-tr/ted_valid_tr.csv"
test_path = "./multitarget-ted/en-tr/ted_test_tr.csv"

train_ds = data_load(train_path, label2id, label_list)
val_ds = data_load(val_path, label2id, label_list)
test_ds = data_load(test_path, label2id, label_list)

punc_ds = DatasetDict({
    "train": train_ds,
    "val": val_ds,
    "test": test_ds,
})

tokenized_punc_ds = punc_ds.map(tokenize_and_align_labels, batched=True, fn_kwargs=({"tokenizer": tokenizer}))
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

EPOCH = 3
compute_metrics = prepare_compute_metrics(label_list)

training_args = TrainingArguments(
            output_dir="./model_save/auto",
            learning_rate=2e-5,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            num_train_epochs=EPOCH,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            disable_tqdm=False,
            log_level="error",
            load_best_model_at_end=True)

optimizer = AdamW(model.parameters(),
                  lr = 1e-3,
                  betas = (0.9, 0.999),
                  eps = 1e-6)

trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_punc_ds["train"],
            eval_dataset=tokenized_punc_ds["val"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None))

start = timeit.default_timer()
trainer.train()
stop = timeit.default_timer()
print(f"Training Time: {stop-start:.2f}s")

curr_time = datetime.now()
print(f"File name: {curr_time}")
model.save_pretrained(f"./model_save/{curr_time}", id2label=id2label, label2id=label2id, num_labels=len(label_list))
tokenizer.save_pretrained(f"./model_save/{curr_time}")

model = BertForTokenClassification.from_pretrained(f"./model_save/{curr_time}", num_labels = len(label_list), id2label = id2label, label2id = label2id, local_files_only=True).to(device)
tokenizer = BertTokenizerFast.from_pretrained(f"./model_save/{curr_time}", local_files_only=True)

test_results = get_test_results(model, tokenizer, label_list, tokenized_punc_ds, device)
print(test_results)

sent_list = []
for sample in test_ds:
    sent_list.append(" ".join(sample["tokens"]))
n_sents = len(sent_list)

pipe = pipeline(task="token-classification", model=model, tokenizer=tokenizer, device=0)
start = timeit.default_timer()
inf_res = pipe(sent_list)
stop = timeit.default_timer()
print(f"Inference Time: {stop-start:.2f}s for {n_sents} sentences")
