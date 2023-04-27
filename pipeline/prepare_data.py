import json
import pandas as pd

TRAIN_PATH = "data/train.json"
DEV_PATH = "data/dev.json"
TEST_PATH = "data/test.json"

###TASK 1
def generate_nli_data(file_path):
    '''
    Generates data from clinical trials for Task 1: Textual entailment (NLI).

    Parameters:
        file_path (str): Path to the JSON of the dataset.

    Returns:
        joint_data: List of training instances in form of "claim [SEP] evidence_text" (str)
        labels: List of labels, either 1 for "Entailment" or 0 for "Contradiction" (int)
    '''

    #Read the file.
    df = pd.read_json(file_path)
    df = df.transpose()

    #Extract claims and labels. Map labels to binary values (0, 1).
    claims = df.Statement.tolist()
    labels = df.Label.tolist()
    labels = map(lambda x : 1 if x == "Entailment" else 0, labels)

    #(Prepare to) Extract all evidence sentences from clinical trials
    evidence_texts = list()
    primary_cts, secondary_cts = df.Primary_id, df.Secondary_id    
    primary_indices = df.Primary_evidence_index 
    secondary_indices = df.Secondary_evidence_index
    sections, types = df.Section_id, df.Type

    #Generate evidence texts for each claim.
    for claim_id in range(len(claims)):
        file_name = "data/CTs/" + primary_cts[claim_id] + ".json"

        with open(file_name, 'r') as f:
            data = json.load(f)
            evidence = "primary trial: " 

            #Evidence for the primary trial is in form:
            # "primary trial: sent_1. sent_2. (...) sent_n."           
            for i in primary_indices[claim_id]:
                evidence += data[sections[claim_id]][i]
                evidence += " "
                
        #If it is a comparative claim, also add evidence sentences from the 2nd trial.
        if types[claim_id] == "Comparison":
            file_name = "data/CTs/" + secondary_cts[claim_id] + ".json"

            #Evidence for the secondary trial is in form:
            # "| secondary trial: sent_1. sent_2. (...) sent_n."
            with open(file_name, 'r') as f:
                data = json.load(f)
                evidence += " | secondary trial: "
                for i in secondary_indices[claim_id]:
                    evidence += data[sections[claim_id]][i]
                    evidence += " "

        evidence_texts.append(evidence)

    #One training instance is: "claim [SEP] full_evidence_text"
    joint_data = list()
    for i in range(len(claims)):
        premise = claims[i]
        hypothesis = evidence_texts[i]
        joint = premise + " [SEP] " + hypothesis
        joint_data.append(joint)

    return joint_data, labels


###TASK 2
def generate_evidence_data(file_path):
    '''
    Generates data from clinical trials for Task 2: Evidence Retrieval (/selection).

    Parameters:
        file_path (str): Path to the JSON of the dataset.

    Returns:
        joint_data: List of training instances in form of "claim [SEP] candidate_sentence" (str)
        labels: List of labels, 0 if candidate_sentence is not evidence, 1 if it is
    '''

    #Read the file.
    df = pd.read_json(file_path)
    df = df.transpose()

    #Extract claims.
    claims = df.Statement.tolist()

    #(Prepare to) Extract all evidence sentences from clinical trials
    primary_cts, secondary_cts = df.Primary_id, df.Secondary_id    
    primary_indices = df.Primary_evidence_index 
    secondary_indices = df.Secondary_evidence_index
    sections, types = df.Section_id, df.Type

    primary_evidence_sentences = list()
    secondary_evidence_sentences = list()

    for idx in range(len(claims)):
        file_name = "data/CTs/" + primary_cts[idx] + ".json"

        #Create a list of all evidence sentences from the primary trial for this claim.
        with open(file_name, 'r') as f:
            data = json.load(f)
            primary_evidence_sentences.append(data[sections[idx]])

        #If it is a comparative claim, also create a list of secondary-trial evidence sentences.
        if types[idx] == "Comparison":
            file_name = "/home/ubuntu/nli4ct/Complete_dataset/CTs/" + secondary_cts[idx] + ".json"

            with open(file_name, 'r') as f:
                data = json.load(f)
                secondary_evidence_sentences.append(data[sections[idx]])
        else:
            secondary_evidence_sentences.append(list())

    #Generate training instances in form of "claim [SEP] candidate_sentence", 
    joint_data = list()

    #Label is 0 if candidate sentece is not evidence for this claim, 1 if it is   
    labels = list() 

    for claim_id in range(len(claims)):
        claim = claims[claim_id]
        primary_sents = primary_evidence_sentences[claim_id]

        for sid in range(len(primary_sents)):
            candidate_sentence = primary_sents[sid]
            j = candidate_sentence + " [SEP] " + claim
            joint_data.append(j)
            labels.append(sid in primary_indices[claim_id])

        if types[claim_id] == "Comparison":
            secondary_sents = secondary_evidence_sentences[claim_id]
            for sid in range(len(secondary_sents)):
                candidate_sentence = secondary_sents[sid]
                j = candidate_sentence + " [SEP] " + claim
                joint_data.append(j)
                labels.append(sid in secondary_indices[claim_id])

        labels = [1 if l else 0 for l in labels]

    return joint_data, labels


