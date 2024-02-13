import sys
sys.path.append('./')
import json
import torch
from rlt.dataset import Dataset, collate_fn
from rlt.models.bicut import BiCut
from rlt.models.choppy import Choppy
from rlt.models.attncut import AttnCut
from rlt.models.mmoecut import MMOECut
from rlt.models.lecut import LeCut
from rlt.trainer import Trainer
from rlt.utils import replicability
import losses
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'
                    )

def training(args):

    dataset = Dataset(args)
    args.feature_dim = dataset.input[0][2].shape[1]
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)

    if args.name == "bicut":
        # hyperparameters reported in the paper
        args.d_lstm = 128
        args.num_lstm_layers = 2
        args.feedforward = 256
        args.lr = 1e-4

        # alpha 0.65 for F1, 10 for NCI; needed to be tuned

        model = BiCut(args)
        loss_function = losses.BiCutLoss(alpha=args.alpha, r=dataset.r)

    elif args.name == "choppy":
        # hyperparameters reported in the paper
        args.num_transformer_layers = 3
        args.nhead = 8
        args.d_transformer = 128
        args.lr = 0.001
        args.batch_size = 64

        model = Choppy(args)
        loss_function = losses.ChoppyLoss()

    elif args.name == "attncut":
        # hyperparameters reported in the paper
        args.d_lstm = 128
        args.num_lstm_layers = 2

        args.d_transformer = 256
        args.nhead = 4
        args.num_transformer_layers = 1
        # args.batch_size = 20/128

        args.lr = 3e-5
        args.tau = 0.95

        model = AttnCut(args)
        loss_function = losses.AttnCutLoss(tau=args.tau)
    elif args.name == "mmoecut":
        # hyperparameters reported in the paper
        args.d_lstm = 128
        args.num_lstm_layers = 2

        args.d_transformer = 256
        args.nhead = 4
        args.num_transformer_layers = 1

        args.num_experts = 3
        args.num_tasks = 3

        args.lr = 3e-5
        args.tau = 0.95
        args.rerank_weight = 0.5
        args.classi_weight = 0.5

        model = MMOECut(args)
        loss_function = losses.MtCutLoss(rerank_weight=args.rerank_weight, classi_weight=args.classi_weight,
                                         num_tasks=args.num_tasks, tau=args.tau)

    elif args.name == "lecut":
        # hyperparameters reported in the paper
        args.d_embd = dataset.d_embd
        args.feature_dim = args.feature_dim - args.d_embd + 1

        args.d_lstm = 112
        args.num_lstm_layers = 2
        args.d_position = 32

        args.d_transformer = 256
        args.nhead = 4
        args.num_transformer_layers = 1

        args.lr = 3e-5
        args.tau = 0.95

        model = LeCut(args)
        # According to the original implementation of LeCut, LeCut uses exactly AttnCutLoss.
        # See: https://github.com/myx666/LeCut/blob/master/Ranked-List-Truncation/utils/losses.py
        loss_function = losses.AttnCutLoss(tau=args.tau)
    else:
        raise NotImplementedError

    if args.warm_up_path is not None:
        logging.info(f"warm up from {args.warm_up_path}")
        model.load_state_dict(torch.load(args.warm_up_path))

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    trainer = Trainer(args, model, loss_function)
    optimizer.zero_grad()

    for epoch_id in range(1, args.epoch_num + 1):
        trainer.training(data_loader, optimizer, epoch_id)
        trainer.save_model(args.checkpoint_path_, epoch_id)


def inference(args):

    dataset = Dataset(args)
    args.feature_dim = dataset.input[0][2].shape[1]
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False)

    if args.name == "bicut":
        # hyperparameters reported in the paper
        args.d_lstm = 128
        args.num_lstm_layers = 2
        args.feedforward = 256
        args.lr = 1e-4

        # alpha 0.65 for F1, 10 for NCI; needed to be tuned

        model = BiCut(args)

    elif args.name == "choppy":
        # hyperparameters reported in the paper
        args.num_transformer_layers = 3
        args.nhead = 8
        args.d_transformer = 128
        args.lr = 0.001
        args.batch_size = 64

        model = Choppy(args)

    elif args.name == "attncut":
        # hyperparameters reported in the paper
        args.d_lstm = 128
        args.num_lstm_layers = 2

        args.d_transformer = 256
        args.nhead = 4
        args.num_transformer_layers = 1
        # args.batch_size = 20/128

        args.lr = 3e-5
        args.tau = 0.95

        model = AttnCut(args)

    elif args.name == "mmoecut":
        # hyperparameters reported in the paper
        args.d_lstm = 128
        args.num_lstm_layers = 2

        args.d_transformer = 256
        args.nhead = 4
        args.num_transformer_layers = 1

        args.num_experts = 3
        args.num_tasks = 3

        args.lr = 3e-5
        args.tau = 0.95
        args.rerank_weight = 0.5
        args.classi_weight = 0.5

        model = MMOECut(args)

    elif args.name == "lecut":
        # hyperparameters reported in the paper
        args.d_embd = dataset.d_embd
        args.feature_dim = args.feature_dim - args.d_embd + 1

        args.d_lstm = 112
        args.num_lstm_layers = 2
        args.d_position = 32

        args.d_transformer = 256
        args.nhead = 4
        args.num_transformer_layers = 1

        args.lr = 3e-5
        args.tau = 0.95

        model = LeCut(args)

    else:
        raise NotImplementedError

    for epoch_id in range(1,args.epoch_num+1):
        logging.info("*"*20)
        logging.info(f"infer epoch {epoch_id}")


        checkpoint_name = args.checkpoint_path_ + str(epoch_id).zfill(3) + '.pkl'
        model.load_state_dict(torch.load(checkpoint_name))
        trainer = Trainer(args, model, None)
        trainer.inference(data_loader, epoch_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, required=True) # specify [bicut, choppy, attncut, mmoecut, lecut]
    parser.add_argument("--infer", action='store_true')

    parser.add_argument("--eet", action='store_true')
    parser.add_argument("--a", type=float)
    parser.add_argument("--b", type=float)

    parser.add_argument("--seq_len", type=int, required=True) # specify
    parser.add_argument("--alpha", type=float) # specify
    parser.add_argument("--tau", type=float)
    parser.add_argument("--d_lstm", type=int)
    parser.add_argument("--d_transformer", type=int)
    parser.add_argument("--nhead", type=int)
    parser.add_argument("--num_lstm_layers", type=int)
    parser.add_argument("--num_transformer_layers", type=int)
    parser.add_argument("--feedforward", type=int)

    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--feature_path", type=str, default='')
    parser.add_argument("--qrels_path", type=str)
    parser.add_argument("--truncation_path", type=str)
    parser.add_argument("--label_path", type=str)

    parser.add_argument("--binarise_qrels", action='store_true') # only turn on for datasets having graded relevance judgments

    parser.add_argument("--epoch_num", type=int, required=True) # specify
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int, required=True) # specify

    parser.add_argument("--clip", type=float, default=1.)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--interval", type=int, default=10)

    args = parser.parse_args()

    args.dataset_class = args.feature_path.split("/")[-3]
    args.dataset_name = args.feature_path.split("/")[-1].split(".")[0]
    args.retriever= "-".join(args.feature_path.split("/")[-1].split(".")[1].split("-")[1:])


    if args.name == "bicut":
        args.metric=f"alpha{args.alpha}" # only impact training
    else:
        args.metric = ".".join(args.label_path.split("/")[-1].split(".")[2:-1])
        assert "-".join(args.label_path.split("/")[-1].split(".")[1].split("-")[1:]) == args.retriever

    # extra setting
    args.param=""
    if args.name == "lecut":
        embedding = "-".join(args.feature_path.split("/")[-1].split(".")[2].split("-")[1:])
        args.param = f"-embed-{embedding}"


    if args.infer is True:
        args.checkpoint_path_ = f"{args.checkpoint_path}/{args.checkpoint_name}/"
        args.setup = f"{args.dataset_name}.{args.retriever}.{args.name}{args.param}-ckpt-{args.checkpoint_name}"
        #args.output_path_ = f"{args.output_path}/{args.dataset_name}.{args.retriever}.{args.name}"
        args.output_path_ = f"{args.output_path}/{args.dataset_name}.{args.retriever}"

        if not os.path.exists(args.output_path_):
            os.makedirs(args.output_path_)

    else:
        # training
        args.setup = f"{args.dataset_name}.{args.retriever}.{args.name}{args.param}.{args.metric}"
        args.checkpoint_path_ = f"{args.checkpoint_path}/{args.setup}/"

    if not os.path.exists(args.checkpoint_path_):
        os.makedirs(args.checkpoint_path_)

    replicability(seed=args.random_seed)

    logging.info(f"Is GPU available? {torch.cuda.is_available()}")
    logging.info("torch_version:{}".format(torch.__version__))
    logging.info("CUDA_version:{}".format(torch.version.cuda))
    logging.info("cudnn_version:{}".format(torch.backends.cudnn.version()))

    if args.infer is True:
        inference(args)
    else:
        training(args)