from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim import AdamW
from datetime import datetime
import tensorflow as tf
from tqdm import tqdm
import seaborn as sns
import torch.nn as nn
import pandas as pd 
import numpy as np
import warnings 
import torch
import gc
import re
import os
warnings.filterwarnings("ignore")


def load_path():
    # Use absolute path to define the file path
    current_script_path = os.path.abspath(__file__)
    amls_dir_path = os.path.dirname(os.path.dirname(current_script_path))
    return amls_dir_path

def data_preprocessing(text):
        text = text.lower()
        text = re.sub(r'https?://www\.\S+\.cm', '', text)
        text = re.sub(r'[^a-zA-Z|\s]', '', text)
        text = re.sub(r'\*+', 'swear', text)
        text = re.sub('<.*?>', '', text) # Remove HTML from text
        text = re.sub(r'(.)\1{3,}', r'\1', text)
        return text


def preprocess_BERT():
    # load data path
    DATA_PATH = load_path()
    TRAIN_PATH = os.path.join(DATA_PATH, 'Datasets')
    # construct file paths
    test_csv_path = os.path.join(TRAIN_PATH, "twitter_validation.csv")
    train_csv_path = os.path.join(TRAIN_PATH, "twitter_training.csv")
    # read data
    test_df = pd.read_csv(test_csv_path, header=None)
    train_df = pd.read_csv(train_csv_path, header=None)
    # init clean and map text data
    test_df.reset_index(drop=True,inplace=True)
    train_df.reset_index(drop=True,inplace=True)
    df1 = pd.concat([train_df, test_df], axis=0)
    df1.drop([0], axis=1, inplace=True)
    df1.columns = ['platform', 'sentiment', 'text']
    df1.drop(['platform'], axis=1, inplace=True)
    df1.sentiment = df1.sentiment.map({"Neutral": 0, "Irrelevant": 0, "Positive": 1, "Negative": 2})
    df1.dropna(inplace=True)
    # create the new dataset
    total_samples = 7500
    counts = df1['sentiment'].value_counts(normalize=True)
    samples_per_class = (counts * total_samples).round().astype(int)  # Ensure integer sample sizes
    # Sampling for each category
    sampled_dfs = []
    for sentiment, samples in samples_per_class.items():  # Use items() instead of iteritems()
        sampled_dfs.append(
            df1[df1['sentiment'] == sentiment].sample(n=samples, random_state=42)
        )
    df = pd.concat(sampled_dfs).reset_index(drop=True)
    df = df.sample(frac=1)
    # Comprehensive clean the text data
    df['text'] = df['text'].astype(str).apply(lambda x:data_preprocessing(x))

    # set the dataset class
    class BERTDataset:
        def __init__(self, texts, sentiments, tokenizer, max_len):
            self.texts = texts
            self.sentiments = sentiments
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, item):
            text = str(self.texts[item])
            sentiment = self.sentiments[item]

            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True,
            )

            return {
                'text': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'sentiment': torch.tensor(sentiment, dtype=torch.long)
            }
    
    # BERT Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")    
    
    # split the dataset into train, valid and test       
    y = df.sentiment.values
    train_df, tem_df = train_test_split(df, test_size = 0.3, random_state=42,stratify = y)
    y_tem = tem_df.sentiment.values
    valid_df, test_df = train_test_split(tem_df, test_size=0.5, random_state=42, stratify=y_tem)

    # set the parameter values
    MAX_LEN = max([len(x.split()) for x in df['text']])
    BATCH_SIZE = 16

    # DataLoader
    def create_data_loader(df, tokenizer, max_len, batch_size):
        ds = BERTDataset(
            texts=df.text.to_numpy(),
            sentiments=df.sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True
        )
    
    train_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
    valid_loader = create_data_loader(valid_df, tokenizer, MAX_LEN, BATCH_SIZE)
    test_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

    return train_loader, valid_loader, test_loader



def model_BERT():
    # define the BERT model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    return model



def train_BERT(model, train_loader, valid_loader):
    # define the train epoch logic
    def train_one_epoch(model, train_loader, optimizer, device):
        # set init values 
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        for data in tqdm(train_loader, desc="Training"):
            input_ids = data["text"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["sentiment"].to(device)
            optimizer.zero_grad()
            output = model(input_ids=input_ids,attention_mask=attention_mask,labels=targets)
            loss = output.loss
            logits = output.logits
            preds = torch.argmax(logits, dim=1)
            correct_predictions = torch.sum(preds == targets)
            accuracy = correct_predictions.double() / targets.size(0)

            loss.backward()
            optimizer.step()
            # update training loss and accuracy
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
        return epoch_loss / len(train_loader), epoch_accuracy / len(train_loader)
    # define the validation epoch logic
    def validate_one_epoch(model, valid_loader, device):
        # set init values
        valid_loss = 0.0
        valid_accuracy = 0.0
        ######################
        # validate the model #
        ######################
        model.eval()
        for data in tqdm(valid_loader,desc="Validation"):
            input_ids = data['text'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = data['sentiment'].to(device)
            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
                loss = output.loss
                logits = output.logits
                preds = torch.argmax(logits, dim=1)
                correct_predictions = torch.sum(preds == targets)
                accuracy = correct_predictions.double() / targets.size(0)
                # update average validation loss and accuracy
                valid_loss += loss.item()
                valid_accuracy += accuracy.item()
        return valid_loss / len(valid_loader), valid_accuracy / len(valid_loader)

    def fit(
            model, epochs, device, optimizer, train_loader, valid_loader=None
        ):
            # set init values
            valid_loss_min = np.Inf
            train_losses = []
            valid_losses = []
            train_accs = []
            valid_accs = []
            # set the logic about epoch
            for epoch in range(1, epochs + 1):
                # clear useless data
                gc.collect()

                print(f"{'='*50}")
                print(f"EPOCH {epoch} - TRAINING...")
                train_loss, train_acc = train_one_epoch(model,
                    train_loader, optimizer, device
                )
                print(
                    f"\n\t[TRAIN] EPOCH {epoch} - LOSS: {train_loss}, ACCURACY: {train_acc}\n"
                )
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                gc.collect()

                if valid_loader is not None:
                    gc.collect()
            
                    print(f"EPOCH {epoch} - VALIDATING...")
                    valid_loss, valid_acc = validate_one_epoch(model,
                        valid_loader, device
                    )
                    print(f"\t[VALID] LOSS: {valid_loss}, ACCURACY: {valid_acc}\n")
                    valid_losses.append(valid_loss)
                    valid_accs.append(valid_acc)
                    gc.collect()

                    # show if validation loss has decreased
                    if valid_loss <= valid_loss_min and epoch != 1:
                        print(
                            "Validation loss decreased ({:.4f} --> {:.4f}).".format(
                                valid_loss_min, valid_loss
                            )
                        )
                    valid_loss_min = valid_loss

            return {
                "train_loss": train_losses,
                "valid_loss": valid_losses,
                "train_acc": train_accs,
                "valid_acc": valid_accs,
            }
    # load path and set init data
    init_path = load_path()
    acc_train = 0
    acc_valid = 0
    ls_train = np.Inf
    ls_valid = np.Inf
    # set parameters
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    # record start time
    print(f"INITIALIZING TRAINING ")
    start_time = datetime.now()
    print(f"Start Time: {start_time}")
    # train model
    logs = fit(
        model=model,
        epochs=10,
        device=device,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )
    print(f"Execution time: {datetime.now() - start_time}")

    # plot the accuracy and loss values of training and validation with each epoch
    fig, ax = plt.subplots(2,1)
    ax[0].plot(logs['train_loss'], color='b', label="Training Loss")
    ax[0].plot(logs['valid_loss'], color='r', label="Validation Loss")
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(logs['train_acc'], color='b', label="Training Accuracy")
    ax[1].plot(logs['valid_acc'], color='r',label="Validation Accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    # save diagram
    plot = os.path.join(init_path, 'plot', 'BERT accuracy and loss diagram')
    plt.savefig(plot)
    plt.close()

    # get max accuracy and min loss
    for items in logs['train_acc']:
        if items > acc_train:
            acc_train = items

    for items in logs['valid_acc']:
        if items > acc_valid:
            acc_valid = items

    for items in logs['train_loss']:
        if items < ls_train:
            ls_train = items
    
    for items in logs['valid_loss']:
        if items < ls_valid:
            ls_valid = items

    return acc_train, acc_valid, ls_train, ls_valid


def evaluate_BERT(model, test_loader):
    # define the evaluate logic
    def evaluate(model, test_loader, device):
        # set init values
        test_loss = 0.0
        test_accuracy = 0.0
        all_preds = []
        all_targets = []
        ######################
        # evaluate the model #
        ######################
        model.eval()
        for data in tqdm(test_loader,desc="Testing"):
            input_ids = data['text'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = data['sentiment'].to(device)
            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
                loss = output.loss
                logits = output.logits
                preds = torch.argmax(logits, dim=1)
                correct_predictions = torch.sum(preds == targets)
                accuracy = correct_predictions.double() / targets.size(0)
                # update average test loss and accuracy
                test_loss += loss.item()
                test_accuracy += accuracy.item()
                # get prediction and true values
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        # calculate average losses
        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)
        return test_loss, test_accuracy, np.array(all_preds), np.array(all_targets)
    
    # get data path
    init_path = load_path()
    # set parameter
    device = torch.device("cpu")
    # evaluate model
    test_loss, test_acc, all_preds, all_targets = evaluate(model, test_loader, device)
    # Calculate confusion matrix
    conf_mat = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(10, 10))
    # plot it
    sns.heatmap(conf_mat, annot=True, fmt='d', ax=ax, cmap="Blues")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    # save diagram
    plot = os.path.join(init_path, 'plot', 'BERT confusion matrix diagram')
    plt.savefig(plot)
    plt.close()

    # make performance metrics for each classes
    accuracy = accuracy_score(all_targets, all_preds)

    # define the values in normal, micro and macro state
    num_classes = 3

    precision = precision_score(all_targets, all_preds, average=None)
    precision_micro = precision_score(all_targets, all_preds, average='micro')
    precision_macro = precision_score(all_targets, all_preds, average='macro')

    recall = recall_score(all_targets, all_preds, average=None)
    recall_micro = recall_score(all_targets, all_preds, average='micro')
    recall_macro = recall_score(all_targets, all_preds, average='macro')

    f1 = f1_score(all_targets, all_preds, average=None)
    f1_micro = f1_score(all_targets, all_preds, average='micro')
    f1_macro= f1_score(all_targets, all_preds, average='macro')

    precision = np.append(precision, [precision_micro, precision_macro])
    recall = np.append(recall, [recall_micro, recall_macro])
    f1 = np.append(f1, [f1_micro, f1_macro])
    classes = list(range(num_classes)) + ['Micro', 'Macro']

    x = np.arange(len(classes))
    width = 0.2

    fig1, ax1 = plt.subplots(figsize=(14, 6))
    rects1 = ax1.bar(x - width, precision, width, label='Precision')
    rects2 = ax1.bar(x, recall, width, label='Recall')
    rects3 = ax1.bar(x + width, f1, width, label='F1')
    # make performance diagram
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics (Sentiment Analysis)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='upper right')
    ax1.grid(True)
    # save diagram
    plot1 = os.path.join(init_path, 'plot', 'BERT Performance Metrics diagram')
    plt.savefig(plot1)
    plt.close()

    return test_acc, test_loss


