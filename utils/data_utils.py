import os
import logging

import torch 
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence

class InputFeatures(object):

    def __init__(self, input_ids):
        self.src_tensor = input_ids[0]
        self.target_tensor = input_ids[1]


class en_fr_processor:
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._read_file(data_dir)


    def _read_file(self, data_dir):
        '''
        read file
        '''
        data = []

        with open(os.path.join(data_dir, "news-commentary-v9.fr-en.en")) as en_file, open(os.path.join(data_dir, "news-commentary-v9.fr-en.fr")) as fr_file:
           for fr_sentence, en_sentence in zip(fr_file, en_file):
             if fr_sentence and en_sentence:
                data.append([en_sentence.strip(), fr_sentence.strip()])
          
        #print(data[:1000])
        return data[:1000]

def convert_examples_to_features(examples,  max_seq_length, encode_method):
    features = []
    for (ex_index, example) in enumerate(examples):

        if not example:
            continue

        token_ids = []
       
        for i, word in enumerate(example):  
            tokens = encode_method(word.strip())
            tokens.insert(0, 1)
            tokens.append(0)
            token_ids.append(tokens)# word token ids   
            #token_ids.extend(tokens)  # all sentence token ids

        if ex_index == 0:
          logging.info("token ids = ")
          logging.info(token_ids)
        logging.debug("token ids = ")
        logging.debug(token_ids)

        if token_ids:
          features.append(
              InputFeatures(input_ids=token_ids))

    return features


def create_dataset(features):
    #print(f'src tensor : {features[1].src_tensor}')
    all_src_tensor = [torch.tensor(f.src_tensor) for f in features]
    all_target_tensor = [torch.tensor(f.target_tensor) for f in features]

    all_src_tensor = pad_sequence(all_src_tensor, batch_first=True)
    all_target_tensor = pad_sequence(all_target_tensor, batch_first=True)
    return TensorDataset(
        all_src_tensor, all_target_tensor)
