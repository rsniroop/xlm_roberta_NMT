from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from model.xlmr_mt import XLMR_Encoder, XLM_Decoder
from utils.train_utils import add_xlmr_args
from utils.data_utils import en_fr_processor, create_dataset, convert_examples_to_features

from tqdm.notebook import tqdm
from tqdm import trange

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser()
    parser = add_xlmr_args(parser)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = en_fr_processor()

    train_examples = processor.get_train_examples(args.data_dir)
    
    # preparing model configs
    hidden_size = 768 if 'base' in args.pretrained_path else 1024 # TODO: move this inside model.__init__

    device = 'cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'

    # creating model
    model = XLMR_Encoder_Decoder(pretrained_path=args.pretrained_path,
                    hidden_size=hidden_size,
                    dropout_p=args.dropout, device=device)

    model.encoder.to(device)
    model.decoder.to(device)
    
    params = model.encoder.named_parameters() + model.decoder.named_parameters()

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params]}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=1, t_total=1)

    train_features = convert_examples_to_features(
        train_examples, args.max_seq_length, model.encoder.encode_word)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    #logger.info("  Num steps = %d", num_train_optimization_steps)

    train_data = create_dataset(train_features)

    train_sampler = RandomSampler(train_data)
    
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    for _ in tqdm(range(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        tbar = tqdm(train_dataloader, desc="Iteration")
            
        model.encoder.train()
        for step, batch in enumerate(tbar):
            batch = tuple(t.to(device) for t in batch)
            src_tensor, target_tensor = batch
            enc_out = model.encoder(src_tensor)
            torch.nn.utils.clip_grad_norm_(
                model.encoder.parameters(), args.max_grad_norm)


            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.encoder.zero_grad()
            

    model.encoder.to(device)

if __name__ == "__main__":
    main()
