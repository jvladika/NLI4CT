import json
import pandas as pd 

TRAIN_DATA = "data/train.json"

def generate_multi_data(file_path):
    df = pd.read_json(file_path)
    df = df.transpose()

    #Extract the claims and NLI labels (Entailment/Contradiction).
    claims = df.Statement.tolist()
    nli_labels = df.Label.tolist()

    primary_indices = df.Primary_evidence_index.tolist()
    secondary_indices = df.Secondary_evidence_index.tolist()
    primary_cts = df.Primary_id.tolist()
    secondary_cts = df.Secondary_id.tolist()
    types = df.Type.tolist()
    sections = df.Section_id.tolist()

    primary_evidence_sentences = list()
    secondary_evidence_sentences = list()

    #Process the clinical trial report files.
    for idx in range(len(claims)):
        file_name = "data/CTs/" + primary_cts[idx] + ".json"

        with open(file_name, 'r') as f:
            data = json.load(f)
            primary_evidence_sentences.append(data[sections[idx]])

        if types[idx] == "Comparison":
            file_name = "data/CTs/" + secondary_cts[idx] + ".json"
            with open(file_name, 'r') as f:
                data = json.load(f)
                secondary_evidence_sentences.append(data[sections[idx]])
        else:
            secondary_evidence_sentences.append(list())

    #Generate the joint data instances and labels of evidence sentences. 
    joint_data = list()
    evidence_labels =  list()
    for claim_id in range(len(claims)):
        claim = claims[claim_id]
        current_evidence_labels = list()

        primary_sents = primary_evidence_sentences[claim_id]
        full_instance = claim + " [SEP] "
        
        #One data instance is a concatenation of the claim and all sentences from the CTR.
        # "claim [SEP] candidate_sent_1 [SEP] candidate_sent_2 (...) [SEP] candidate_sent_n"
        for sid in range(len(primary_sents)):
            candidate_sentence = primary_sents[sid]
            full_instance += candidate_sentence
            full_instance += " [SEP] "
            current_evidence_labels.append(1 if sid in primary_indices[claim_id] else 0)

        if types[claim_id] == "Comparison":        
            secondary_sents = secondary_evidence_sentences[claim_id]
            for sid in range(len(secondary_sents)):
                candidate_sentence = secondary_sents[sid]
                full_instance += candidate_sentence
                full_instance += " [SEP] "
                current_evidence_labels.append(1 if sid in secondary_indices[claim_id] else 0)
        joint_data.append(full_instance)

        #For a given clinical trial, evidence string is a string of 0's and 1's,
        # denoting whether the i-th candidate sentence is or is not evidence.
        evidence_string = ""
        for lab in current_evidence_labels:
            evidence_string += str(int(lab))
        evidence_labels.append(evidence_string)

    return joint_data, nli_labels, evidence_labels

    import torch

class CtDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, stances):
        self.encoded = encodings
        self.labels = labels
        self.stances = stances

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encoded.items()}
        item['labels'] = self.labels[idx]
        item['stance'] = self.stances[idx]
        return item

    def __len__(self):
        return len(self.labels)     
  