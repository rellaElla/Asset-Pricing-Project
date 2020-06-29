

# Used Imports

# System Imports
import os
import sys
from pathlib import Path
import pickle

# Usual Stuff
import numpy as np
import pandas as pd
import time

# Data Processing
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# NN Stuff
from torch import nn, optim
import torch

# NLP Imports
from keras.preprocessing.sequence import pad_sequences
import transformers
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertConfig, get_linear_schedule_with_warmup
from preprocess import preprocess
import datetime

def load_in(path):
    df: pd.DataFrame = pd.read_csv(path, encoding='latin-1')
    df['text'] = preprocess(df['text'])

    return df

if __name__=='__main__':
    # Set path
    sys.path.append(Path(os.path.join(os.path.abspath(''), '.../')).resolve().as_posix())

    # Find dataset and preprocess it 
    dataset_path: Path = Path('.../ticker_data/sentiment_data/aapl_sentiment.csv').resolve()

    df: pd.DataFrame = load_in(dataset_path)

    tweets: pd.DataFrame = df.text.values
    labels: pd.DataFrame = df.label.values

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

    input_ids: list = []

    for tweet in tweets:
        encoded_sent = tokenizer.encode(tweet, 
                                        add_special_tokens=True, 
                                        max_length=128, 
                                        return_tensors='pt')

        input_ids.append(encoded_sent.flatten())

    print('Original: ', tweets[0])
    print('Token IDs: ', input_ids[0])

    print('Max tweet length: ', max([len(tweet) for tweet in input_ids]))

    MAX_LEN: int = int(np.mean([len(tweet) for tweet in input_ids]))

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', value=0,
                                                        truncating='post', padding='post')

    attention_masks: list = []

    for tweet in input_ids:
        mask: list = [int(token_id > 0) for token_id in tweet]

        attention_masks.append(mask)

    attention_masks: np.ndarray = np.array(attention_masks)

    RANDOM_STATE = 1
    batch_size = 128

    # Split tweets, labels, and masks into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(input_ids, labels, random_state = RANDOM_STATE, test_size = 0.1)
    train_masks, val_masks, _, _ = train_test_split(attention_masks, labels, random_state = RANDOM_STATE, test_size = 0.1)

    # Convert to PyTorch Tensors
    X_train: torch.Tensor = torch.tensor(X_train, dtype=torch.int64).cuda()
    y_train: torch.Tensor = torch.tensor(y_train, dtype=torch.float32).cuda()
    train_masks: torch.Tensor = torch.tensor(train_masks, dtype=torch.int64).cuda()
    
    X_val: torch.Tensor = torch.tensor(X_val, dtype=torch.int64).cuda()
    y_val: torch.Tensor = torch.tensor(y_val
    , dtype=torch.float32).cuda()
    val_masks: torch.Tensor = torch.tensor(val_masks, dtype=torch.int64).cuda()
    
    # Create PyTorch Dataset and Loader 
    train_data = TensorDataset(X_train, train_masks, y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(X_val, val_masks, y_val)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=1, output_attentions=False, output_hidden_states=False)
    model.cuda()

    params = list(model.named_parameters())
    print(f'Model has {len(params)} parameters')

    # Set AdamW Optimizer
    eta=1e-5
    epsilon=1e-8
    epochs=1

    optimizer = AdamW(model.parameters(), lr=eta, eps=epsilon)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat=labels.flatten()
        return np.sum(pred_flat==labels_flat)/len(labels_flat)

    def format_time(elapsed):
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    # Begin Training
    for epoch_i in range(epochs):
        print('======== Epoch {:} / {:} ======='.format(epoch_i + 1, epochs))

        model.train()
        total_loss: float = 0

        t0: float = time.time()
        loss_values: list = []
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed: str = format_time(time.time() - t0)

                print('Batch {:>5} of {:>5,}. Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids: torch.Tensor = batch[0]
            b_input_mask: torch.Tensor = batch[1]
            b_labels: torch.Tensor = batch[2]

            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            loss: torch.Tensor = outputs[0]
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            avg_train_loss = total_loss/len(train_dataloader)

            loss_values.append(avg_train_loss)
            print('Average Training Loss: {0:.2f}'.format(avg_train_loss))
            print('Training epoch took: {:}'.format(format_time(time.time() - t0)))


        # Begin Validation
        print('Running Validation...')
        t0: float = time.time()
        
        # Change mode
        model.eval()
        eval_loss, eval_accuracy = 0, 0

        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in val_dataloader:
            b_input_ids, b_input_mask, b_labels = batch
            
            with torch.no_grad():
                outputs: torch.Tensor = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

                logits: float = outputs[0]
                logits = logits.detach().cpu().numpy()
                label_ids: np.ndarray = b_labels.cpu().numpy()

                tmp_eval_accuracy: float = flat_accuracy(logits, label_ids)

                eval_accuracy+=tmp_eval_accuracy
                nb_eval_steps+=1
                print('Accuracy: {0:.2f}'.format(eval_accuracy/nb_eval_steps))
            
        print('Validation took: {:}'.format(format_time(time.time() - t0)))
    
    print('Training Complete')



    






  