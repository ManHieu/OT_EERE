from dataclasses import dataclass, field
from typing import List, Optional

import transformers


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated list of dataset names, for training."}
    )

    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to data directory"}
    )
    
    tokenizer: str = field(
        default = None,
        metadata= {"help": "The tokenizer used to prepare data"}
    )

    encoder: str = field(
        default = '/vinai/hieumdt/pretrained_models/models/roberta-base',
        metadata= {"help": "The path of encoder model"}
    )

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, shorter sequences will be padded."
        },
    )

    batch_size: int = field(
        default = 8,
        metadata= {"help": "Batch size"}
    )
    
    n_fold: int = field(
        default = 1,
        metadata={"help": "Number folds of dataset"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Arguments for the Trainer.
    """
    output_dir: str = field(
        default='experiments',
        metadata={"help": "The output directory where the results and model weights will be written."}
    )

    seed: int = field(
        default = 1741,
        metadata= {"help": "Number labels in this dataset"}
    )

    num_labels: int = field(
        default = None,
        metadata= {"help": "Number labels in this dataset"}
    )
    
    loss_weights: List[float] = field(
        default = None,
        metadata= {"help": "Weight for Cross-entropy loss"}
    )

    regular_loss_weight: float = field(
        default =  0.1,
        metadata= {"help": "Weight of regularization loss"}
    )

    OT_loss_weight: float = field(
        default =  0.1,
        metadata= {"help": "Weight of OT loss (cost of transportation plan)"}
    )

    max_seq_len: int = field(
        default = 512,
        metadata= {"help": "Max lengh of sequence counted by tokens"}
    )

    encoder_lr: float = field(
        default=5e-5,
        metadata={"help": "Learning rate of encoder"}
    )

    lr: float = field(
        default=5e-4,
        metadata={"help": "Learning rate of other layers"}
    )

    num_epoches : int = field(
        default=5,
        metadata={"help": "number pretrain epoches"}
    )



@dataclass 
class ModelArguments:
    encoder_name_or_path: str = field(
        default = '/vinai/hieumdt/pretrained_models/models/roberta-base',
        metadata= {"help": "The path of encoder model"}
    )

    distance_emb_size: int = field(
        default = 0,
        metadata= {"help": "Embedding size of distance to triggers"}
    )

    gcn_outp_size: int = field(
        default = 512,
        metadata= {"help": "Output size of GCN"}
    )

    gcn_num_layers: int = field(
        default = 5,
        metadata= {"help": "Number layer of convolution layers in GCN"}
    )

    hidden_size: int = field(
        default = 0,
        metadata= {"help": "hidden layer size"}
    )

    residual_type: str = field(
        default = 'concat',
        metadata= {"help": "Type of residual connection"}
    )

    rnn_num_layers: int = field(
        default = 1,
        metadata= {"help": "LSTM number layer in GCN"}
    )

    dropout: float = field(
        default = 0.5,
        metadata= {"help": "Dropput rate"}
    )

    OT_eps: float = field(
        default = 0.1,
        metadata= {"help": "Regularization coefficient in OT"}
    )

    OT_max_iter: int = field(
        default = 100,
        metadata= {"help": "Number of iteraction steps in OT"}
    )

    OT_reduction: str = field(
        default = 'mean',
        metadata= {"help": "Type of reduction to compute cost in OT"}
    )

    fn_actv: str = field(
        default = 'relu',
        metadata= {"help": "Active function of classifier"}
    )
    




