import sys
sys.path.append('./')
import json
import os
from collections import defaultdict
import torch
import csv
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'
                    )

class Trainer(object):
    def __init__(self, args, model, loss_function):
        super(Trainer, self).__init__()
        self.args = args


        if torch.cuda.is_available():
            self.model = model.cuda()
            if loss_function:
                self.loss_function = loss_function.cuda()
        else:
            self.model = model
            if loss_function:
                self.loss_function = loss_function


    def training(self, data_loader, optimizer, epoch_id):
        self.model.train()

        step = 0
        loss_display = 0

        for j, data in enumerate(data_loader, 0):
            if torch.cuda.is_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                data = data_cuda

            p= self.model(data) # [B, S, 1], [B, S, 2], [[B, S, 1], [B, S, 1], [B, S, 1]]
            loss = self.loss_function(p, data)

            loss_display += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % self.args.interval == 0:

                logging.info(f'Training: {self.args.setup}')
                logging.info(f'Epoch:{epoch_id}, Step:{step}, Loss:{loss_display/self.args.interval}, LR:{self.args.lr}')

                loss_display = 0


    def save_model(self, checkpoint_path, epoch_id):
        torch.save(self.model.state_dict(), os.path.join(checkpoint_path, '.'.join([str(epoch_id).zfill(3), 'pkl'])))
        logging.info("Saved the model trained on epoch {} ".format(epoch_id))


    def inference(self, data_loader, epoch_id):
        self.model.eval()

        with torch.no_grad():
            q2p_w = open(self.args.output_path_ + "/" + self.args.setup + "-" + str(epoch_id).zfill(3), 'w')

            for index, data in enumerate(data_loader, 0):
                if (index+1) % self.args.interval ==0 or (index+1)==1:
                    logging.info("{}: doing {} / total {} in epoch {}".format(self.args.setup, index+1, len(data_loader), str(epoch_id).zfill(3)))

                if torch.cuda.is_available():
                    data_cuda = dict()
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data_cuda[key] = value.cuda()
                        else:
                            data_cuda[key] = value
                    data = data_cuda

                p = self.model(data)

                if self.args.name == 'bicut':
                    p = np.argmax(p.detach().cpu().numpy(), axis=2) # [B, S]
                    k = []
                    for example in p:
                        if np.sum(example) == self.args.seq_len:
                            k.append(self.args.seq_len)
                        else:
                            # if an item is predicted as truncation, which means that this item is not relevant enough
                            k.append(np.argmin(example))
                            # or k.append(np.argmin(example)+1)


                elif self.args.name =="mmoecut":
                    p = p[-1].detach().cpu().squeeze(-1).numpy()  # [B, S]
                    k = np.argmax(p, axis=1) + 1  # [B]
                else:
                    p = p.detach().cpu().squeeze(-1).numpy()  # [B, S]
                    k = np.argmax(p, axis=1) + 1  # [B]

                logging.info(f"Predicted k: {k}")

                for index, qid in enumerate(data["qid"]):
                    q2p_w.write(qid + '\t' + str(k[index]) + '\n')

            q2p_w.close()

