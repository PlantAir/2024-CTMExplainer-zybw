from random import random
from time import time
from typing import Union
from pandas import DataFrame
from pathlib import Path
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from ctmexplainer.xgraph.method.base_explainer_tg import BaseExplainerTG, BaseExplainerCTMS
from ctmexplainer.xgraph.evaluation.metrics_tg_utils import fidelity_inv_tg
from ctmexplainer.xgraph.method.tg_score import _set_tgat_data
from ctmexplainer.xgraph.models.ext.tgat.module import TGAN
from ctmexplainer.xgraph.models.ext.tgn.model.tgn import TGN
import random as rd
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau


# def _create_explainer_input(model: Union[TGAN, TGN], model_name, all_events, candidate_events=None, event_idx=None, device=None):
#     # DONE: explainer input should have both the target event and the event that we want to assign a weight to.
#
#     if model_name in ['tgat', 'tgn']:
#         event_idx_u, event_idx_i, event_idx_t = _set_tgat_data(all_events, event_idx)
#         event_idx_new = event_idx
#         t_idx_u_emb = model.node_raw_embed[ torch.tensor(event_idx_u, dtype=torch.int64, device=device), : ]
#         t_idx_i_emb = model.node_raw_embed[ torch.tensor(event_idx_i, dtype=torch.int64, device=device), : ]
#         t_idx_t_emb = model.time_encoder( torch.tensor(event_idx_t, dtype=torch.float32, device=device).reshape((1, -1)) ).reshape((1, -1))
#         t_idx_e_emb = model.edge_raw_embed[ torch.tensor([event_idx_new, ], dtype=torch.int64, device=device), : ]
#
#         target_event_emb = torch.cat([t_idx_u_emb,t_idx_i_emb, t_idx_t_emb, t_idx_e_emb ], dim=1)
#
#
#         candidate_events_u, candidate_events_i, candidate_events_t = _set_tgat_data(all_events, candidate_events)
#         candidate_events_new = candidate_events
#
#         candidate_u_emb = model.node_raw_embed[ torch.tensor(candidate_events_u, dtype=torch.int64, device=device), : ]
#         candidate_i_emb = model.node_raw_embed[ torch.tensor(candidate_events_i, dtype=torch.int64, device=device), : ]
#         candidate_t_emb = model.time_encoder( torch.tensor(candidate_events_t, dtype=torch.float32, device=device).reshape((1, -1)) ).reshape((len(candidate_events_t), -1))
#         candidate_e_emb = model.edge_raw_embed[ torch.tensor(candidate_events_new, dtype=torch.int64, device=device), : ]
#
#         candidate_events_emb = torch.cat([candidate_u_emb, candidate_i_emb, candidate_t_emb, candidate_e_emb], dim=1)
#
#         # 将candiadte_events_emb随机选择2-10个，组成sequence gru的输入，然后假设剩余的n个candiadte_events_emb，剩余的candiadte_events_emb组成新candiadte_events_emb
#         # 随机选择2-10个候选事件
#         num_selected = np.random.randint((2, min(10, candidate_events_emb.shape[0]))
#         selected_indices = random.sample(range(candidate_events_emb.shape[0]), num_selected)
#         selected_events_emb = candidate_events_emb[selected_indices]
#
#
#         # # 剩余的候选事件组成新的candidate_events_emb
#         mask = torch.ones(candidate_events_emb.size(0), dtype=torch.bool, device=device)
#         mask[selected_indices] = False
#         # new_candidate_events_emb = candidate_events_emb[mask]
#         # input_seq = selected_events_emb.repeat(new_candidate_events_emb.shape[0], 1)
#         input_seq = selected_events_emb.repeat(candidate_events_emb.shape[0], 1)
#
#         return input_seq, candidate_events_emb, target_event_emb, mask
#     else:
#         raise NotImplementedError


import torch
import torch.nn as nn


class ctms_model(nn.Module):
    def __init__(self, input_size, hidden_size, mlp_input_size, mlp_hidden_size, gru_layers=1, max_len=20):
        super(ctms_model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_layers = gru_layers
        self.mlp_input_size = mlp_input_size
        self.max_len = max_len
        self.gru = nn.GRU(input_size, hidden_size, gru_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size, hidden_size, gru_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + 2 * mlp_input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, 1)
        )
        # self.mlp = nn.Sequential(
        #         nn.Linear(2*mlp_input_size, mlp_hidden_size),
        #         nn.ReLU(),
        #         nn.Linear(mlp_hidden_size, 1)
        #     )
        self.max_len = max_len

    def forward(self, sequence, additional_vector, target_embed):
        batch_size = sequence.size(0)
        seq_len = sequence.size(1)

        # Zero padding
        # print(sequence.shape)
        # print(self.max_len)
        # if seq_len < self.max_len:
        #     padding = torch.zeros((batch_size, self.max_len - seq_len, self.input_size), device=sequence.device)
        #     # padding = padding.repeat(batch_size, 1)
        #     # print(padding.shape)
        #     # print(sequence.shape)
        #     sequence = torch.cat((padding, sequence), dim=1)
        # print(sequence.shape)
        # print(sequence.shape)
        # GRU representation
        # print(sequence)

        gru_output, _ = self.gru(sequence)
        last_hidden_state = gru_output[:, -1, :]

        # lstm_output, (hidden, cell) = self.lstm(sequence)  # LSTM返回最后一层的隐藏状态和单元状态
        # last_hidden_state = hidden[-1]

        # print(last_hidden_state)
        # print(target_embed.shape)
        # print(last_hidden_state.shape)
        # print(additional_vector.shape)
        # Concatenate with additional vector

        combined = torch.cat((target_embed, last_hidden_state, additional_vector), dim=1)
        # combined = torch.cat((target_embed, additional_vector), dim=1)

        # print(target_embed)
        # print('==========')
        # print(additional_vector)

        # MLP output
        output = torch.sigmoid(self.mlp(combined))
        # output = self.mlp(combined)
        return output


class CTMs(BaseExplainerCTMS):
    def __init__(self,
                 model,
                 model_name: str,
                 explainer_name: str,
                 dataset_name: str,
                 all_events: DataFrame,
                 explanation_level: str,
                 device,
                 verbose: bool = True,
                 results_dir=None,
                 debug_mode=True,
                 threshold_num=25,
                 # specific params for PGExplainerExt
                 train_epochs: int = 50,
                 explainer_ckpt_dir=None,
                 reg_coefs=None,
                 batch_size=64,
                 lr=1e-2
                 ):
        super(CTMs, self).__init__(model=model,
                                   model_name=model_name,
                                   explainer_name=explainer_name,
                                   dataset_name=dataset_name,
                                   all_events=all_events,
                                   explanation_level=explanation_level,
                                   device=device,
                                   verbose=verbose,
                                   results_dir=results_dir,
                                   debug_mode=debug_mode,
                                   threshold_num=threshold_num)
        self.train_epochs = train_epochs
        self.explainer_ckpt_dir = explainer_ckpt_dir
        self.reg_coefs = reg_coefs
        self.batch_size = batch_size
        self.lr = lr
        self.expl_input_dim = None
        self._init_explainer()

        self.explainer_ckpt_path = self._ckpt_path(self.explainer_ckpt_dir, self.model_name, self.dataset_name,
                                                   self.explainer_name)
        # if exists, load. Otherwise train.
        # print(str(self.explainer_ckpt_path))
        if self.explainer_ckpt_path.exists():
            state_dict = torch.load(self.explainer_ckpt_path)
            self.explainer_model.load_state_dict(state_dict)
            print(f'explainer ckpt loaded from {str(self.explainer_ckpt_path)}')
        else:
            print(f'explainer ckpt not found at {str(self.explainer_ckpt_path)}')
            print('start training...')
            self._train()
            print('training finished')

    @staticmethod
    # def _create_explainer(model, model_name, device, max_len=20):
    #     if model_name == 'tgat':
    #         expl_input_dim = model.model_dim * 2  # 2 * (dim_u + dim_i + dim_t + dim_e)
    #     elif model_name == 'tgn':
    #         expl_input_dim = model.n_node_features * 2
    #     else:
    #         raise NotImplementedError
    #     explainer_model = ctms_model(expl_input_dim, 128, expl_input_dim, 64, max_len=max_len)
    #
    #     explainer_model = explainer_model.to(device)
    #     return explainer_model

    def _create_explainer(model, model_name, device):
        if model_name == 'tgat':
            expl_input_dim = model.model_dim * 8 # 2 * (dim_u + dim_i + dim_t + dim_e)
        elif model_name == 'tgn':
            expl_input_dim = model.n_node_features * 8
        else:
            raise NotImplementedError

        explainer_model = nn.Sequential(
            nn.Linear(int(expl_input_dim*11), 128),
            nn.ReLU(),
            nn.Linear(128, 1),  ##### version 1
            # nn.Sigmoid(), ##### version 2
        )
        explainer_model = explainer_model.to(device)
        return explainer_model, expl_input_dim


    @staticmethod
    def _ckpt_path(ckpt_dir, model_name, dataset_name, explainer_name, epoch=None):
        if epoch is None:
            return Path(ckpt_dir) / f'{model_name}_{dataset_name}_{explainer_name}_ctm1_expl_ckpt.pt'
        else:
            return Path(ckpt_dir) / f'{model_name}_{dataset_name}_{explainer_name}_ctm1_expl_ckpt_ep{epoch}.pt'

    def _init_explainer(self):
        self.explainer_model, self.input_size = self._create_explainer(self.model, self.model_name, self.device)

    def __call__(self, node_idxs: Union[int, None] = None, event_idxs: Union[int, None] = None):
        CTMS = []
        for i, event_idx in enumerate(event_idxs):
            print(f'\nexplain {i}-th: {event_idx}')
            self._initialize(event_idx)

            tick = time()
            CTM = self.explain(event_idx=event_idx)
            tock = time() - tick
            # results_list.append([list(candidate_weights.keys()), list(candidate_weights.values())])
            self._save_candidate_scores(CTM, event_idx, len(CTM) * [tock])
            CTMS.append(CTM)
        # import ipdb; ipdb.set_trace()
        return CTMS

    def _tg_predict(self, event_idx, CTM_index, use_explainer=False):
        if self.model_name in ['tgat', 'tgn']:
            src_idx_l, target_idx_l, cut_time_l = _set_tgat_data(self.all_events, event_idx)
            edge_weights = None
            if use_explainer:
                rest_events = [item for index, item in enumerate(self.candidate_events) if index not in CTM_index]
                current_CTM = list(np.array(self.candidate_events)[CTM_index])
                # print(rest_events)
                # print(current_CTM)

                # candidate_events_new = _set_tgat_events_idxs(self.candidate_events) # these temporal edges to alter attn weights in tgat
                input_seq, candidate_events_emb, target_event_emb = _create_explainer_input(self.model, self.model_name, self.all_events,  \
                                                     sequence=current_CTM, candidate_events=rest_events, event_idx=event_idx,
                                                     device=self.device)
                # import ipdb; ipdb.set_trace()
                # print(len(input_seq))
                seq_len = len(input_seq[0])
                # print(input_seq)
                # print(input_seq.size())
                if seq_len < 20:
                    padding = torch.zeros((len(rest_events), 20 - seq_len, input_seq.size(2)), device=input_seq.device)
                    # print(padding)
                    # print(padding.size())
                    # padding = padding.repeat(20, 1)
                    # print(padding.shape)
                    # print(sequence.shape)
                    input_seq = torch.cat((padding, input_seq), dim=1)
                # print(input_seq.size())
                # input_seq = torch.cat((input_seq), dim=1)
                input_seq = input_seq.view(input_seq.size(0), input_seq.size(1)*input_seq.size(2))

                # print(input_seq.size())
                combined = torch.cat((target_event_emb, input_seq, candidate_events_emb), dim=-1)

                edge_weights = self.explainer_model(combined)
                candidate_weights_dict = {
                    'candidate_events': torch.tensor(rest_events, dtype=torch.int64, device=self.device),
                    'edge_weights': edge_weights,
                    }
                max_value, max_index = torch.max(edge_weights, dim=0)
                best_item = rest_events[max_index.item()]
                best_index = self.candidate_events.index(best_item)
                CTM_index.append(best_index)
            else:
                candidate_weights_dict = None
            # NOTE: use the 'src_ngh_eidx_batch' in module to locate mask fill positions
            output = self.model.get_prob(src_idx_l, target_idx_l, cut_time_l, logit=True,
                                         candidate_weights_dict=candidate_weights_dict)
            return output, edge_weights, CTM_index

        else:
            raise NotImplementedError

    def _loss(self, masked_pred, original_pred, prev_pred):
        # TODO: improve the loss?
        # Explanation loss
        if prev_pred > 0:  # larger better
            error_loss = original_pred * (prev_pred / masked_pred)
        else:
            error_loss = - original_pred * (prev_pred / masked_pred)

        return error_loss

    def _obtain_train_idxs(self, ):
        size = 500
        if self.dataset_name in ['wikipedia', 'reddit']:
            train_e_idxs = np.random.randint(int(len(self.all_events) * 0.05), int(len(self.all_events) * 0.95), (size,))
            train_e_idxs = shuffle(train_e_idxs)  # TODO: not shuffle?
        elif self.dataset_name in ['simulate_v1', 'simulate_v2']:
            # positive_indices = self.all_events.label == 1
            # pos_events = self.all_events[positive_indices].e_idx.values
            # train_e_idxs = np.random.choice(pos_events, size=size, replace=False)
            train_e_idxs = np.random.choice(self.all_events.e_idx.values, size=size, replace=False)

        return train_e_idxs

    def _train(self, ):
        self.explainer_model.train()
        optimizer = torch.optim.Adam(self.explainer_model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        # train_e_idxs = np.random.randint(int(len(self.all_events)*0.2), int(len(self.all_events)*0.6), (2000, )) # NOTE: set train event idxs
        min_loss = float('inf')  # 初始化最小loss为无穷大
        best_state_dict = None  # 初始化最佳模型状态
        for e in range(self.train_epochs):
            train_e_idxs = self._obtain_train_idxs()

            optimizer.zero_grad()
            loss = torch.tensor([0], dtype=torch.float32, device=self.device)
            loss_list = []
            counter = 0
            skipped_num = 0


            for i, event_idx in tqdm(enumerate(train_e_idxs), total=len(train_e_idxs), desc=f'epoch {e}'):  # training
                self._initialize(event_idx)  # NOTE: needed
                if len(self.candidate_events) == 0 or len(self.candidate_events) == 0:  # skip bad samples
                    skipped_num += 1
                    continue
                # get the outputs of the target model for the given events
                CTM_index = []
                previous_index = []
                for j in range(len(self.candidate_events)):
                    # get the soft-masked outputs of the target model
                    # soft-masks are computed by the explainer model (which is called navigator in the paper)
                    original_pred, mask_values_, CTM_index = self._tg_predict(event_idx, CTM_index, use_explainer=False)
                    if len(previous_index) == 0:
                        previous_pred = original_pred
                    else:
                        previous_pred, previous_values, _ = self._tg_predict(event_idx, previous_index,
                                                                             use_explainer=True)
                    previous_index = copy.deepcopy(CTM_index)
                    masked_pred, mask_values, CTM_index = self._tg_predict(event_idx, CTM_index, use_explainer=True)

                    # binary cross entropy between the original and masked outputs
                    id_loss = self._loss(masked_pred, original_pred, previous_pred)
                    # import ipdb; ipdb.set_trace()
                    id_loss = id_loss.flatten()
                    assert len(id_loss) == 1
                    loss += id_loss

                loss = loss / (len(self.candidate_events))
                loss_list.append(loss.cpu().detach().item())
                loss.backward()
                optimizer.step()
                loss = torch.tensor([0], dtype=torch.float32, device=self.device)
                optimizer.zero_grad()

        #     state_dict = self.explainer_model.state_dict()
        #     torch.save(state_dict, self.explainer_ckpt_path)
        #     tqdm.write(
        #         f"epoch {e} loss epoch {np.mean(loss_list)}, skipped: {skipped_num}, ckpt saved: {self.explainer_ckpt_path}")
        #
        # state_dict = self.explainer_model.state_dict()
        # torch.save(state_dict, self.explainer_ckpt_path)
        # print('train finished')
        # print(f'explainer ckpt saved at {str(self.explainer_ckpt_path)}')
            # import ipdb; ipdb.set_trace()
            # state_dict = self.explainer_model.state_dict()
            #
            # torch.save(state_dict, self.explainer_ckpt_path)
            # tqdm.write(
            #     f"epoch {e} loss epoch {np.mean(loss_list)}, skipped: {skipped_num}, ckpt saved: {self.explainer_ckpt_path}")
            epoch_loss = np.mean(loss_list)
            scheduler.step(epoch_loss)
            torch.save(self.explainer_model.state_dict(), self.explainer_ckpt_dir+f'/saved_epochs/{self.model_name}_{self.dataset_name}_{self.explainer_name}_epoch{e}')
            if epoch_loss < min_loss:  # 如果当前epoch的loss更低，则保存模型
                min_loss = epoch_loss
                best_state_dict = self.explainer_model.state_dict()
                torch.save(best_state_dict, self.explainer_ckpt_path)
                tqdm.write(
                    f"epoch {e} loss epoch {epoch_loss}, skipped: {skipped_num}, ckpt saved: {self.explainer_ckpt_path}")
            else:
      
                tqdm.write(
                    f"epoch {e} loss epoch {epoch_loss}, skipped: {skipped_num}")

            # 保存最终的模型
        torch.save(best_state_dict, self.explainer_ckpt_path)
        state_dict = self.explainer_model.state_dict()
        torch.save(state_dict, self.explainer_ckpt_path)
        print('train finished')
        print(f'explainer ckpt saved at {str(self.explainer_ckpt_path)}')


    def explain(self, node_idx=None, event_idx=None):
        self.explainer_model.eval()
        CTM_index = []
        print(len(self.candidate_events))
        for j in range(len(self.candidate_events)):
            # get the soft-masked outputs of the target model
            # soft-masks are computed by the explainer model (which is called navigator in the paper)
            masked_pred, mask_values, CTM_index = self._tg_predict(event_idx, CTM_index, use_explainer=True)
            # print(CTM_index)
        CTM = list(np.array(self.candidate_events)[CTM_index])
        return CTM

    def expose_explainer_model(self):
        return self.explainer_model  # torch.nn.Sequential


import torch
import numpy as np
from tgnnexplainer.xgraph.method.attn_explainer_tg import AttnExplainerTG
from tgnnexplainer.xgraph.method.other_baselines_tg import PGExplainerExt
from tgnnexplainer.xgraph.method.tg_score import _set_tgat_data
from pandas import DataFrame
from copy import deepcopy


class PGNavigator():
    """
        Navigator class implementing the author's version of the navigator.
        When called, it computes
        - the importance scores of the candidate events
        - the aggregated attention scores of the candidate events,
          masked by the importance scores
        - the final candidate scores are the aggregated attention scores
    """

    def __init__(self,
                 model,
                 model_name: str,
                 explainer_name: str,
                 dataset_name: str,
                 all_events: DataFrame,
                 explanation_level: str,
                 device,
                 verbose: bool = True,
                 results_dir=None,
                 debug_mode=True,
                 train_epochs: int = 50,
                 explainer_ckpt_dir=None,
                 reg_coefs=None,
                 batch_size=64,
                 lr=1e-4):
        """
            Create a PgExplainerExt instance and call expose_explainer_model on it.
            Return the exposed explainer model, which is the MLP.

            Note: the exposed explainer model is a torch.nn.Sequential object.
            Note: the decision whether to load the parameters or to pre-train the explainer
                is made in the constructor of the PGExplainerExt class.
        """
        self.model = model
        self.model_name = model_name
        self.device = device
        self.all_events = all_events
        # handles loading/training the MLP
        ctms_model = CTMs(model=model,
                          model_name=model_name,
                          explainer_name=explainer_name,
                          dataset_name=dataset_name,
                          all_events=all_events,
                          explanation_level=explanation_level,
                          device=device,
                          verbose=verbose,
                          results_dir=results_dir,
                          debug_mode=debug_mode,
                          train_epochs=train_epochs,
                          explainer_ckpt_dir=explainer_ckpt_dir,
                          reg_coefs=reg_coefs,
                          batch_size=batch_size,
                          lr=lr
                          )
        # deepcopy the exposed explainer model, we will discard the PGExplainerExt instance
        self.mlp = deepcopy(ctms_model.expose_explainer_model())

    def __call__(self, candidate_event_idx, target_idx):
        """
            Construct input for the pre-trained navigator (MLP)
            Call the navigator (MLP) on the input
            Evaluate the target on the candidate events, masked by the output of the navigator
            Return the mean attention scores over the layers of the target model
        """
        # ensure evaluation mode
        self.mlp.eval()
        input_expl = _create_explainer_input(self.model, self.model_name, self.all_events,
                                             candidate_events=candidate_event_idx, event_idx=target_idx,
                                             device=self.device)

        # compute importance scores
        edge_weights = self.mlp(input_expl)

        # added to original model attention scores
        candidate_weights_dict = {
            'candidate_events': torch.tensor(candidate_event_idx, dtype=torch.int64, device=self.device),
            'edge_weights': edge_weights,
        }
        src_idx_l, target_idx_l, cut_time_l = _set_tgat_data(
            self.all_events, target_idx)
        # run forward pass on the target model with soft-masks applied to the input events
        output = self.model.get_prob(
            src_idx_l, target_idx_l, cut_time_l, logit=True, candidate_weights_dict=candidate_weights_dict)
        # obtain aggregated attention scores for the masked candidate input events
        e_idx_weight_dict = AttnExplainerTG._agg_attention(
            self.model, self.model_name)
        # final edge weights are the aggregated attention scores masked by the pre-trained navigator
        edge_weights = np.array([e_idx_weight_dict[e_idx]
                                 for e_idx in candidate_event_idx])
        # added to original model attention scores

        return edge_weights


class CTMSNavigator(PGNavigator):
    """
        Our implementation of the navigator.
        When called, it computes
        - the importance scores of the candidate events
        - these scores are used as candidate weights
    """

    def __call__(self, sequence_event_idx, candidate_event_idx, target_idx):
        """
            Construct input for the pre-trained navigator (MLP)
            Call the navigator (MLP) on the input
            Return the edge weights

            Note: the input consists of pair-wise concatenation
                of the target event and the candidate events.
        """
        # mlp = tips
        input_expl, cand_emb, target_event_emb = _create_explainer_input(self.model, self.model_name,
                                                                         self.all_events, \
                                                                         candidate_events=candidate_event_idx,
                                                                         event_idx=target_idx,
                                                                         device=self.device, sequence=sequence_event_idx)
        # import ipdb; ipdb.set_trace()
        edge_weights = self.mlp(input_expl, cand_emb, target_event_emb)

        return edge_weights


# def _create_explainer_input(model: Union[TGAN, TGN], model_name, all_events, candidate_events=None, event_idx=None,
#                             device=None):
#     # DONE: explainer input should have both the target event and the event that we want to assign a weight to.
#
#     if model_name in ['tgat', 'tgn']:
#         event_idx_u, event_idx_i, event_idx_t = _set_tgat_data(all_events, event_idx)
#         # event_idx_new = _set_tgat_events_idxs(event_idx)
#         event_idx_new = event_idx
#         # t_idx_u_emb = model.node_raw_embed[torch.tensor(event_idx_u, dtype=torch.int64, device=device), :]
#         # t_idx_i_emb = model.node_raw_embed[torch.tensor(event_idx_i, dtype=torch.int64, device=device), :]
#         # import ipdb; ipdb.set_trace()
#         t_idx_t_emb = model.time_encoder(
#             torch.tensor(event_idx_t, dtype=torch.float32, device=device).reshape((1, -1))).reshape((1, -1))
#         t_idx_e_emb = model.edge_raw_embed[torch.tensor([event_idx_new, ], dtype=torch.int64, device=device), :]
#
#         # target_event_emb = torch.cat([t_idx_u_emb, t_idx_i_emb, t_idx_t_emb, t_idx_e_emb], dim=1)
#         target_event_emb = torch.cat([t_idx_t_emb, t_idx_e_emb], dim=1)
#
#
#         candidate_events_u, candidate_events_i, candidate_events_t = _set_tgat_data(all_events, candidate_events)
#         candidate_events_new = candidate_events
#
#         # candidate_u_emb = model.node_raw_embed[torch.tensor(candidate_events_u, dtype=torch.int64, device=device), :]
#         # candidate_i_emb = model.node_raw_embed[torch.tensor(candidate_events_i, dtype=torch.int64, device=device), :]
#         candidate_t_emb = model.time_encoder(
#             torch.tensor(candidate_events_t, dtype=torch.float32, device=device).reshape((1, -1))).reshape(
#             (len(candidate_events_t), -1))
#         candidate_e_emb = model.edge_raw_embed[torch.tensor(candidate_events_new, dtype=torch.int64, device=device), :]
#
#         # candiadte_events_emb = torch.cat([candidate_u_emb, candidate_i_emb, candidate_t_emb, candidate_e_emb], dim=1)
#         candiadte_events_emb = torch.cat([candidate_t_emb, candidate_e_emb], dim=1)
#         input_expl = torch.cat([target_event_emb.repeat(candiadte_events_emb.shape[0], 1), candiadte_events_emb], dim=1)
#         # import ipdb; ipdb.set_trace()
#         return input_expl
#
#     else:
#         raise NotImplementedError


def _create_explainer_input(model: Union[TGAN, TGN], model_name, all_events, sequence=None, candidate_events=None,
                            event_idx=None,
                            device=None):
    # DONE: explainer input should have both the target event and the event that we want to assign a weight to.

    if model_name in ['tgat', 'tgn']:
        event_idx_u, event_idx_i, event_idx_t = _set_tgat_data(all_events, event_idx)
        event_idx_new = event_idx
        t_idx_u_emb = model.node_raw_embed[torch.tensor(event_idx_u, dtype=torch.int64, device=device), :]
        t_idx_i_emb = model.node_raw_embed[torch.tensor(event_idx_i, dtype=torch.int64, device=device), :]
        t_idx_t_emb = model.time_encoder(
            torch.tensor(event_idx_t, dtype=torch.float32, device=device).reshape((1, -1))).reshape((1, -1))
        t_idx_e_emb = model.edge_raw_embed[torch.tensor([event_idx_new, ], dtype=torch.int64, device=device), :]

        target_event_emb = torch.cat([t_idx_u_emb, t_idx_i_emb, t_idx_t_emb, t_idx_e_emb], dim=1)
        # target_event_emb = torch.cat([t_idx_t_emb, t_idx_e_emb], dim=1)
        target_event_emb = torch.clamp(target_event_emb, min=-0.99, max=0.99)
        # print(target_event_emb)
        # print(candidate_events)
        candidate_events_u, candidate_events_i, candidate_events_t = _set_tgat_data(all_events, candidate_events)
        candidate_events_new = candidate_events

        candidate_u_emb = model.node_raw_embed[torch.tensor(candidate_events_u, dtype=torch.int64, device=device), :]
        candidate_i_emb = model.node_raw_embed[torch.tensor(candidate_events_i, dtype=torch.int64, device=device), :]
        candidate_t_emb = model.time_encoder(
            torch.tensor(candidate_events_t, dtype=torch.float32, device=device).reshape((1, -1))).reshape(
            (len(candidate_events_t), -1))
        candidate_e_emb = model.edge_raw_embed[torch.tensor(candidate_events_new, dtype=torch.int64, device=device), :]

        candidate_events_emb = torch.cat([candidate_u_emb, candidate_i_emb, candidate_t_emb, candidate_e_emb], dim=1)
        # candidate_events_emb = torch.cat([candidate_t_emb, candidate_e_emb], dim=1)
        candidate_events_emb = torch.clamp(candidate_events_emb, min=-0.99, max=0.99)

        sequence = list(sequence)
        if len(sequence) == 0:
            sequence = [event_idx]
        # print(sequence)
        sequence_events_u, sequence_events_i, sequence_events_t = _set_tgat_data(all_events, sequence)
        sequence_events_new = sequence
        sequence_u_emb = model.node_raw_embed[torch.tensor(sequence_events_u, dtype=torch.int64, device=device), :]
        sequence_i_emb = model.node_raw_embed[torch.tensor(sequence_events_i, dtype=torch.int64, device=device), :]
        sequence_t_emb = model.time_encoder(
            torch.tensor(sequence_events_t, dtype=torch.float32, device=device).reshape((1, -1))).reshape(
            (len(sequence_events_t), -1))
        sequence_e_emb = model.edge_raw_embed[torch.tensor(sequence_events_new, dtype=torch.int64, device=device), :]
        sequence_events_emb = torch.cat([sequence_u_emb, sequence_i_emb, sequence_t_emb, sequence_e_emb], dim=1)

        # sequence_events_emb = torch.cat([sequence_t_emb, sequence_e_emb], dim=1)
        sequence_events_emb = torch.clamp(sequence_events_emb, min=-0.99, max=0.99)
        # print(candidate_events_emb.shape)
        input_seq = sequence_events_emb.repeat(candidate_events_emb.shape[0], 1, 1)
        target_event_emb = target_event_emb.repeat(candidate_events_emb.shape[0], 1)

        return input_seq, candidate_events_emb, target_event_emb
    else:
        raise NotImplementedError

#
# import torch
# import numpy as np
# from tgnnexplainer.xgraph.method.attn_explainer_tg import AttnExplainerTG
# from tgnnexplainer.xgraph.method.other_baselines_tg import PGExplainerExt
# from tgnnexplainer.xgraph.method.tg_score import _set_tgat_data
# from pandas import DataFrame
# from copy import deepcopy
#
#
# class PGNavigator():
#     """
#         Navigator class implementing the author's version of the navigator.
#         When called, it computes
#         - the importance scores of the candidate events
#         - the aggregated attention scores of the candidate events,
#           masked by the importance scores
#         - the final candidate scores are the aggregated attention scores
#     """
#
#     def __init__(self,
#                  model,
#                  model_name: str,
#                  explainer_name: str,
#                  dataset_name: str,
#                  all_events: DataFrame,
#                  explanation_level: str,
#                  device,
#                  verbose: bool = True,
#                  results_dir=None,
#                  debug_mode=True,
#                  train_epochs: int = 50,
#                  explainer_ckpt_dir=None,
#                  reg_coefs=None,
#                  batch_size=64,
#                  lr=1e-4):
#         """
#             Create a PgExplainerExt instance and call expose_explainer_model on it.
#             Return the exposed explainer model, which is the MLP.
#
#             Note: the exposed explainer model is a torch.nn.Sequential object.
#             Note: the decision whether to load the parameters or to pre-train the explainer
#                 is made in the constructor of the PGExplainerExt class.
#         """
#         self.model = model
#         self.model_name = model_name
#         self.device = device
#         self.all_events = all_events
#         # handles loading/training the MLP
#         pg_explainer = PGExplainerExt(model=model,
#                                       model_name=model_name,
#                                       explainer_name=explainer_name,
#                                       dataset_name=dataset_name,
#                                       all_events=all_events,
#                                       explanation_level=explanation_level,
#                                       device=device,
#                                       verbose=verbose,
#                                       results_dir=results_dir,
#                                       debug_mode=debug_mode,
#                                       train_epochs=train_epochs,
#                                       explainer_ckpt_dir=explainer_ckpt_dir,
#                                       reg_coefs=reg_coefs,
#                                       batch_size=batch_size,
#                                       lr=lr
#                                       )
#         # deepcopy the exposed explainer model, we will discard the PGExplainerExt instance
#         self.mlp = deepcopy(pg_explainer.expose_explainer_model())
#
#     def __call__(self, candidate_event_idx, target_idx):
#         """
#             Construct input for the pre-trained navigator (MLP)
#             Call the navigator (MLP) on the input
#             Evaluate the target on the candidate events, masked by the output of the navigator
#             Return the mean attention scores over the layers of the target model
#         """
#         # ensure evaluation mode
#         self.mlp.eval()
#         input_expl = _create_explainer_input(self.model, self.model_name, self.all_events,
#                                              candidate_events=candidate_event_idx, event_idx=target_idx,
#                                              device=self.device)
#
#         # compute importance scores
#         edge_weights = self.mlp(input_expl)
#
#         # added to original model attention scores
#         candidate_weights_dict = {
#             'candidate_events': torch.tensor(candidate_event_idx, dtype=torch.int64, device=self.device),
#             'edge_weights': edge_weights,
#             }
#         src_idx_l, target_idx_l, cut_time_l = _set_tgat_data(
#             self.all_events, target_idx)
#         # run forward pass on the target model with soft-masks applied to the input events
#         output = self.model.get_prob(
#             src_idx_l, target_idx_l, cut_time_l, logit=True, candidate_weights_dict=candidate_weights_dict)
#         # obtain aggregated attention scores for the masked candidate input events
#         e_idx_weight_dict = AttnExplainerTG._agg_attention(
#             self.model, self.model_name)
#         # final edge weights are the aggregated attention scores masked by the pre-trained navigator
#         edge_weights = np.array([e_idx_weight_dict[e_idx]
#                                  for e_idx in candidate_event_idx])
#         # added to original model attention scores
#
#         return edge_weights
#
#
# class TIPSNavigator(PGNavigator):
#     """
#         Our implementation of the navigator.
#         When called, it computes
#         - the importance scores of the candidate events
#         - these scores are used as candidate weights
#     """
#     def __call__(self, sequence_event_idx, candidate_event_idx, target_idx):
#         """
#             Construct input for the pre-trained navigator (MLP)
#             Call the navigator (MLP) on the input
#             Return the edge weights
#
#             Note: the input consists of pair-wise concatenation
#                 of the target event and the candidate events.
#         """
#         # mlp = tips
#         self.mlp.eval()
#         input_expl = _create_explainer_input(self.model, self.model_name, self.all_events, sequence=sequence_event_idx,
#                     candidate_events=candidate_event_idx, event_idx=target_idx, device=self.device)
#
#         # compute importance scores
#         edge_weights = self.mlp(input_expl)
#
#         return edge_weights

