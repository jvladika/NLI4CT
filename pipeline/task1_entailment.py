import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from prepare_data import generate_nli_data

TRAIN_PATH = "data/train.json"
DEV_PATH = "data/dev.json"
TEST_PATH = "data/test.json"

#Torch dataset used in the models. Consists of encodings of training instances and of labels.
#One training instance is: BERT_TOKENIZER("claim [SEP] evidence_text").
class CtDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


models = ["ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli",
"ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli",
"MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
"microsoft/deberta-v2-xlarge-mnli",
"MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"]

#Compute the metrics (accuracy, precision, recall, F1) for a give prediction.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    return {"accuracy": acc, "precision" : prec, "recall" : recall, "f1": f1}

#Training loop.
def train(model_name):
    #model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

    #Load the models. Adjust max instance length to fit your machine.
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=1024)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                 num_labels=2, ignore_mismatched_sizes=True)

    #Generate joint claim+[SEP]+evidence data.
    joint_train, labels_train = generate_nli_data(TRAIN_PATH)
    joint_dev, labels_dev= generate_nli_data(DEV_PATH)

    #Tokenize the data.    
    encoded_train = tokenizer(joint_train, return_tensors='pt',
                         truncation_strategy='only_first', add_special_tokens=True, padding=True)
    encoded_dev = tokenizer(joint_dev, return_tensors='pt',
                         truncation_strategy='only_first', add_special_tokens=True, padding=True)
   
    #Convert data into datasets
    train_dataset = CtDataset(encoded_train, labels_train)
    dev_dataset = CtDataset(encoded_dev, labels_dev)

    #Define the batch size to fit your GPU memory.
    batch_size = 16

    logging_steps = len(train_data["claims"]) // batch_size
    output_name = f"finetuned-model"

    training_args = TrainingArguments(output_dir=output_name,
                                 per_device_train_batch_size=batch_size,
                                 per_device_eval_batch_size=batch_size,
                                 
                                 #for faster training time
                                 dataloader_pin_memory=True, 
                                 dataloader_num_workers=4,
                                 gradient_accumulation_steps=2,
                                 fp16=True,

                                 #training hyperparameters
                                 num_train_epochs=5,
                                 learning_rate=5e-6,
                                 weight_decay=0.01,
                                 warmup_ratio=0.06,

                                 #other parameters
                                 evaluation_strategy="epoch",
                                 save_strategy="no",
                                 disable_tqdm=False,
                                 logging_steps=logging_steps,
                                 push_to_hub=False)

    trainer = Trainer(model=model, args=training_args,
                    compute_metrics=compute_metrics,
                    train_dataset=train_dataset,
                    eval_dataset=dev_dataset,
                    tokenizer=tokenizer)

    #Start the training process.
    trainer.train()

    #Save the fine-tuned NLI (textual entailment) model.
    trainer.save_model("model-nli")