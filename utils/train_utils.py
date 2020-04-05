from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from seqeval.metrics import f1_score, classification_report
import torch
import torch.nn.functional as F


def add_xlmr_args(parser):
     """
     Adds training and validation arguments to the passed parser
     """

     parser.add_argument("--data_dir",
                         default=None,
                         type=str,
                         required=True,
                         help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
     parser.add_argument("--pretrained_path", default=None, type=str, required=True,
                         help="pretrained XLM-Roberta model path")
     parser.add_argument("--output_dir",
                         default=None,
                         type=str,
                         required=True,
                         help="The output directory where the model predictions and checkpoints will be written.")
     # Other parameters
     parser.add_argument("--cache_dir",
                         default="",
                         type=str,
                         help="Where do you want to store the pre-trained models downloaded from s3")
     parser.add_argument("--max_seq_length",
                         default=128,
                         type=int,
                         help="The maximum total input sequence length after WordPiece tokenization. \n"
                              "Sequences longer than this will be truncated, and sequences shorter \n"
                              "than this will be padded.")
     parser.add_argument("--train_batch_size",
                         default=32,
                         type=int,
                         help="Total batch size for training.")
     parser.add_argument("--learning_rate",
                         default=5e-5,
                         type=float,
                         help="The initial learning rate for Adam.")
     parser.add_argument("--num_train_epochs",
                         default=3,
                         type=int,
                         help="Total number of training epochs to perform.")
     parser.add_argument("--weight_decay", default=0.01, type=float,
                         help="Weight deay if we apply some.")
     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                         help="Epsilon for Adam optimizer.")
     parser.add_argument("--max_grad_norm", default=1.0, type=float,
                         help="Max gradient norm.")
     parser.add_argument("--no_cuda",
                         action='store_true',
                         help="Whether not to use CUDA when available")
     parser.add_argument('--seed',
                         type=int,
                         default=42,
                         help="random seed for initialization")
     parser.add_argument('--gradient_accumulation_steps',
                         type=int,
                         default=1,
                         help="Number of updates steps to accumulate before performing a backward/update pass.")
     parser.add_argument('--loss_scale',
                         type=float, default=0,
                         help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                              "0 (default value): dynamic loss scaling.\n"
                              "Positive power of 2: static loss scaling value.\n")
     parser.add_argument('--dropout', 
                         type=float, default=0.3,
                         help = "training dropout probability")
     
     return parser
