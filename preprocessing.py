"""
DALI 2024W Application - Machine Learning Track
John Guerrerio

This code contains parsing and preprocessing code for the LitCovid NER task.  Given a PubTator file (see README for file
format explanation), parses it, tokenizes and preprocesses the text, and saves the results in a json that can be read by
transformers for model training/inference.
"""

import stanza
import json


def LitCovid_preprocessing(input_file, outfile, verbose=False):
    '''
    Given a PubTator file, generates labels, performs initial preprocessing/tokenization, and saves the results to a json
    which can be read by the transformers library

    Arguments:
        input_file: A string representing the file name or path to the PubTator file containing LitCovid annotations
        outfile: A string representing the name of the json file that will contain the preprocessed documents; will be created if it doesn't exist
        verbose: A boolean representing weather extra information should be printed as the function runs (useful for debugging; defaults to False)
    Returns:
        A json file containing the preprocessed documents and their corresponding labels
    Raises:
        None
    '''
    nlp = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'}, package='none')  # tokenizer

    with open(input_file) as f:
        s = f.readlines()

    documents = []  # list of dictionaries for each document
    labels = []  # entities, their character spans, and entity type
    text = ""  # combined title and abstract
    title = ""  # title of paper
    abstract = ""  # abstract text
    PMID = 0  # identifying number for each paper

    for sent in s:  # iterate over all lines of the given PubTator file
        length_PMID = 0
        if sent != "\n":  # Finds PMID given that PMIDs have variable length
            for char in sent:
                if char.isdigit():
                    length_PMID += 1
                else:
                    break
            PMID = sent[0:length_PMID].strip()

        # If a line is either a title or an abstract (not a label)
        if len(sent) >= 5 and ("|t|" in sent[0:15] or "|a|" in sent[0:15]):
            text += sent[length_PMID + 3:].strip() + " "  # drops PMID that begins each line
            if "|t|" in sent[0:15]:
                title += sent
            if "|a|" in sent[0:15]:
                abstract += sent
        elif len(sent) >= 5:  # Otherwise the line must be a label
            # the tab character separates the different label fields in PubTator files
            label = sent[length_PMID + 1:].split("\t")
            multi_labels = []  # list of labels where the words the label is describing have been tokenized
            if label[0] == '':
                label.pop(0)
            if " " or "-" in label[2]:  # Handles entities that consist of multiple tokens
                tokens = nlp(label[2]).to_dict()
                cumulative_length = int(label[0])  # where in the text the tokens we're interested in start
                for sentence in tokens:  # spacy returns lists of lists so we have to iterate twice
                    for token in sentence:
                        if token["text"] != "-":  # we want to exclude dashes from the labels
                            # new label for tokenized label
                            new_label = [str(token['start_char'] + cumulative_length),
                                         str(token["end_char"] + cumulative_length), token["text"], label[3]]
                            multi_labels.append(new_label)
            else:
                multi_labels.append(label)

            for label in multi_labels:  # Formats entries in label and removes special characters
                for i, s in enumerate(label):
                    label[i] = label[i].strip()
                    label[i] = ''.join(filter(str.isalnum, label[i]))
                labels.append(label)  # appends the label(s) for a PubTator line to the overall labels list

        if sent == "\n":  # Determines when the end of a title/abstract pair is reached
            tokenized_text = nlp(text).to_dict()  # tokenize text with spacy
            document_labels = []
            sentences = []
            bio_labels = []

            for sentence in tokenized_text:  # Generates labels for a document
                for token in sentence:
                    labeled = False
                    for label in labels:
                        # check for named entities
                        if str(token["start_char"]) == label[0] and str(token["end_char"]) == label[1]:
                            labeled = True
                            if label[3].lower() == "vaccine":  # Labels for the LitCovid NER task
                                document_labels.append(1)
                            if label[3].lower() == "strain":
                                document_labels.append(2)
                            if label[3].lower() == "vaccinefunder":
                                document_labels.append(3)
                    if not labeled:  # non-entity token
                        document_labels.append(0)
                    sentences.append(token["text"])

            # Now we need to generate BIO Labels from the IO labels

            # adds first label to the bio_labels list so when we look back we always have a label to compare to
            if document_labels[0] == 2:  # "up-shift" IO labels to the appropriate B-entity label
                bio_labels.append(3)
            elif document_labels[0] == 3:
                bio_labels.append(5)
            else:
                bio_labels.append(document_labels[0])

            # convert adjacent entity labels to BIO labels
            for i in range(1, len(document_labels)):
                if document_labels[i] == 0:
                    bio_labels.append(0)
                # if a label is the same as the one before it, we need to convert to BIO labels
                # we don't modify document_labels so this comparison is valid
                elif document_labels[i] == document_labels[i - 1]:
                    if document_labels[i] == 1:
                        bio_labels.append(2)
                    if document_labels[i] == 2:
                        bio_labels.append(4)
                    if document_labels[i] == 3:
                        bio_labels.append(6)
                # "up-shift" IO labels to the appropriate B-entity label
                elif document_labels[i] == 2:
                    bio_labels.append(3)
                elif document_labels[i] == 3:
                    bio_labels.append(5)
                else:
                    bio_labels.append(document_labels[i])

            # dictionary for a single document
            documents.append(
                {"PMID": PMID, "tokens": tokenized_text, "sentences": sentences, "full_text": text, "title": title,
                 "abstract": abstract, "ner_tags": document_labels,
                 "bio_ner_tags": bio_labels})  # Formats title/abstract information into a dictionary to be written to a json

            if verbose:  # Useful when debugging
                print("PMID: " + PMID)
                print("Text: " + text)
                print("Title: " + title.strip())
                print("Abstract: " + abstract.strip())
                print("Labels: " + str(labels))
                print("Sentences: " + str(sentences))
                print("Document labels: " + str(document_labels))
                print("BIO Labels: " + str(bio_labels))
                print("\n \n")

            # Reset per-document variables
            labels = []
            text = ""
            title = ""
            abstract = ""

    if verbose:
        print("Total number of documents processed: " + str(len(documents)))

    # Writes documents to a json in a format that can be loaded into a dataset by datasets
    with open(outfile, "w") as o:
        data = {"records": documents}
        json.dump(data, o)


def inference_preprocessing(input_file, outfile, verbose=False):
    """
    Given a PubTator file, performs initial preprocessing/tokenization and saves the results to a json
    which can be read by the transformers library.  Doesn't generate labels as this function is designed to
    preprocess for inference

    Args:
        input_file: Path to the file in PubTator format to perform inference on
        outfile: json file to write preprocessed data to for inference; will be created if it doesn't exist
        verbose: A boolean representing weather extra information should be printed as the function runs (useful for debugging; defaults to false)
    Returns:
        A json file containing the preprocessed documents for inference
    Raises:
        None
    """

    with open(input_file) as f:
        s = f.readlines()

    documents = []  # list of dictionaries for each document
    text = ""  # combined title and abstract
    title = ""  # title of paper
    abstract = ""  # abstract text
    PMID = 0  # identifying number for each paper

    for sent in s:  # iterate over lines of PubTator file
        length_PMID = 0
        if sent != "\n":  # Finds PMID given that PMIDs have variable length
            for char in sent:
                if char.isdigit():
                    length_PMID += 1
                else:
                    break
            PMID = sent[0:length_PMID].strip()

        # If a line is either a title or an abstract (not a label)
        if len(sent) >= 5 and ("|t|" in sent[0:15] or "|a|" in sent[0:15]):
            text += sent[length_PMID + 3:].strip() + " "  # drops PMID that begins each line
            if "|t|" in sent[0:15]:
                title += sent
            if "|a|" in sent[0:15]:
                abstract += sent

        if sent == "\n":  # newline signals the end of a document
            # Records document PMID, full text, title, and abstract
            documents.append({"PMID": PMID, "Text": text, "Title": title, "Abstract": abstract})

            if verbose:
                print("PMID: " + PMID)
                print("Text: " + text)
                print("Title: " + title.strip())
                print("Abstract: " + abstract.strip())

            # Reset per-document variables
            text = ""
            title = ""
            abstract = ""

    if verbose:
        print("Total number of documents processed: " + str(len(documents)))

    # Writes documents to a json in a format that can be loaded into a dataset by datasets
    with open(outfile, "w") as o:
        data = {"records": documents}
        json.dump(data, o)


if __name__ == "__main__":
    LitCovid_preprocessing("LitCovid_Val_Test.PubTator.txt", "LitCovid_combined.json", verbose=True)
