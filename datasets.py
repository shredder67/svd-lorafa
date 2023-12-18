import csv

import torch
import pandas as pd
from torch.utils.data import Dataset


class GLUEDatasetRoberta(Dataset):
    def __init__(self, file, tokenizer, benchmark, max_length=128) -> None:

        assert benchmark.lower() in ['mrpc', 'cola', 'qnli', 'rte', 'sst2', 'stsb', 'qqp', 'mnli']

        self.benchmark = benchmark.lower()        

        kwargs = {'padding': 'max_length',
                  'truncation': True,
                  'max_length': max_length,
                  'return_tensors': 'pt'
                  }

        if self.benchmark == 'mrpc':
            label_column = 'Quality'
            self.csv = pd.read_csv(file, delimiter='\t', quoting=csv.QUOTE_NONE)
            text1 = list(self.csv['#1 String'])
            text2 = list(self.csv['#2 String'])
            self.texts = [text1, text2]

        elif self.benchmark == 'cola':
            label_column = 1
            self.csv = pd.read_csv(file, delimiter='\t', header=None)
            text = self.csv[3]
            self.texts = [list(text)]

        elif self.benchmark == 'qnli' or self.benchmark == 'rte':
            label_column = 'label'
            self.csv = pd.read_csv(file, delimiter='\t', on_bad_lines='skip') 
            self.csv.dropna(inplace=True)
            self.csv.reset_index(inplace=True)
            self.csv[label_column][self.csv[label_column].str.contains('not')] = 0
            self.csv[label_column][self.csv[label_column] != 0] = 1
            self.csv[label_column] = self.csv[label_column].astype(int)
            if self.benchmark == 'qnli':
                text1 = list(self.csv['question'])
                text2 = list(self.csv['sentence'])
            else:
                text1 = list(self.csv['sentence1'])
                text2 = list(self.csv['sentence2'])
            self.texts = [text1, text2]
        
        elif self.benchmark == 'sst2':
            label_column = 'label'
            self.csv = pd.read_csv(file, delimiter='\t')
            text = self.csv['sentence']
            self.texts = [list(text)]

        elif self.benchmark == 'stsb': #MSELoss
            label_column = 'score'
            self.csv = pd.read_csv(file, delimiter='\t', on_bad_lines='skip')
            self.csv[label_column] = self.csv[label_column].astype(float)
            self.csv.dropna(inplace=True)
            self.csv.reset_index(inplace=True)
            text1 = list(self.csv['sentence1'])
            text2 = list(self.csv['sentence2'])
            self.texts = [text1, text2]

        elif self.benchmark == 'qqp':
            label_column = 'is_duplicate'
            self.csv = pd.read_csv(file, delimiter='\t')
            text1 = list(self.csv['question1'])
            text2 = list(self.csv['question2'])
            self.texts = [text1, text2]

        elif self.benchmark == 'mnli': # MultiClass
            label_column = 'label1'
            self.csv = pd.read_csv(file, delimiter='\t', on_bad_lines='skip')
            self.csv.dropna(inplace=True)
            self.csv.reset_index(inplace=True)
            self.csv[label_column][self.csv[label_column]=='neutral'] = 0
            self.csv[label_column][self.csv[label_column]=='entailment'] = 1
            self.csv[label_column][self.csv[label_column]=='contradiction'] = 2
            self.csv[label_column] = self.csv[label_column].astype(int)
            text1 = list(self.csv['sentence1'])
            text2 = list(self.csv['sentence2'])
            self.texts = [text1, text2]

        self.encoded_data = tokenizer(*self.texts, **kwargs)
        self.input_ids = self.encoded_data['input_ids']
        self.attention_mask = self.encoded_data['attention_mask']
        self.labels = torch.tensor(self.csv[label_column])

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]
    
    def __len__(self):
        return len(self.csv)
