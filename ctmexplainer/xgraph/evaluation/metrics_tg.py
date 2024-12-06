from ctypes import Union
from fileinput import filename
from typing import List
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from pathlib import Path
from ctmexplainer.xgraph.evaluation.metrics_tg_utils import fidelity_inv_tg, sparsity_tg


class BaseEvaluator():
    def __init__(self, model_name: str, explainer_name: str, dataset_name: str,
                 explainer: BaseExplainerTG = None,
                 results_dir=None,
                 threshold_num=20,
                 ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.explainer_name = explainer_name

        self.explainer = explainer

        self.results_dir = results_dir
        self.suffix = None
        self.threshold_num = threshold_num

    @staticmethod
    def _save_path(results_dir,
                   model_name,
                   dataset_name,
                   explainer_name,
                   event_idxs,
                   suffix=None,
                   threshold_num=20):
        if isinstance(event_idxs, int):
            event_idxs = [event_idxs, ]

        if suffix is not None:
            filename = Path(
                results_dir) / f'{model_name}_{dataset_name}_{explainer_name}_{event_idxs[0]}_to_{event_idxs[-1]}_eval_{suffix}_th{threshold_num}.csv'
        else:
            filename = Path(
                results_dir) / f'{model_name}_{dataset_name}_{explainer_name}_{event_idxs[0]}_to_{event_idxs[-1]}_eval_th{threshold_num}.csv'
        return filename

    def _save_value_results(self, event_idxs, value_results, suffix=None):
        """save to a csv for plotting"""
        filename = self._save_path(self.results_dir, self.model_name, self.dataset_name, self.explainer_name,
                                   event_idxs, suffix, self.threshold_num)

        df = DataFrame(value_results)
        df.to_csv(filename, index=False)

        print(f'evaluation value results saved at {str(filename)}')

    def _evaluate_one(self, single_results, event_idx):
        raise NotImplementedError

    def evaluate(self, explainer_results, event_idxs):
        event_idxs_results = []
        sparsity_results = []
        fid_inv_results = []
        fid_inv_best_results = []
        fid_var_results = []
        fid_var_best_results = []

        print('\nevaluating...')
        for i, (single_results, event_idx) in enumerate(zip(explainer_results, event_idxs)):
            print(f'\nevaluate {i}th: {event_idx}')
            self.explainer._initialize(event_idx)
            if len(self.explainer.candidate_events) < 6:
                continue

            sparsity_list, fid_inv_list, fid_inv_best_list, fid_var_list, fid_var_best_list = self._evaluate_one(
                single_results, event_idx)

            # import ipdb; ipdb.set_trace()
            event_idxs_results.extend([event_idx] * len(sparsity_list))
            sparsity_results.extend(sparsity_list)
            fid_inv_results.extend(fid_inv_list)
            fid_inv_best_results.extend(fid_inv_best_list)
            fid_var_results.extend(fid_var_list)
            fid_var_best_results.extend(fid_var_best_list)

        results = {
            'event_idx': event_idxs_results,
            'sparsity': sparsity_results,
            'fid_inv': fid_inv_results,
            'fid_inv_best': fid_inv_best_results,
            'fid_var': fid_var_results,
            'fid_var_best': fid_var_best_results,
        }

        self._save_value_results(event_idxs, results, self.suffix)
        return results


class EvaluatorCTMS(BaseEvaluator):
    def __init__(self, model_name: str, explainer_name: str, dataset_name: str,
                 explainer: AttnExplainerTG,
                 results_dir=None,
                 threshold_num=25,
                 ) -> None:
        super(EvaluatorCTMS, self).__init__(model_name=model_name,
                                               explainer_name=explainer_name,
                                               dataset_name=dataset_name,
                                               results_dir=results_dir,
                                               explainer=explainer,
                                               threshold_num=threshold_num
                                               )
        # self.explainer = explainer

    # SOLVED: why 0 in the first row of results csv? sparsity calculation is wrong
    def _evaluate_one(self, single_results, event_idx):
        CTMS = single_results

        candidate_events = self.explainer.candidate_events
        candidate_num = len(candidate_events)
        assert len(CTMS) == candidate_num

        fid_inv_list = []
        fid_var_list = []
        sparsity_list = np.arange(0, 1.05, 0.05)
        for spar in sparsity_list:
            num = int(spar * candidate_num)
            important_events = CTMS[:num + 1]
            b_i_events = self.explainer.base_events + important_events
            # print(f"==============  +++++++++++++b_i   ==============")
            # print(len(important_events))
            # print(important_events)
            # print(len(b_i_events))
            # print(b_i_events)

            important_pred = self.explainer.tgnn_reward_wraper._compute_gnn_score(b_i_events, event_idx)
            ori_pred = self.explainer.tgnn_reward_wraper.original_scores

            fid_inv = fidelity_inv_tg(ori_pred, important_pred)
            fid_inv_list.append(fid_inv)

            # calculate data varying
            important_np = np.array(important_events)
            candidate_np = np.array(candidate_events)
            rest = self.explainer.base_events + list(candidate_np[~np.isin(candidate_np, important_np)])
            rest_pred = self.explainer.tgnn_reward_wraper._compute_gnn_score(rest, event_idx)
            fid_var = fidelity_inv_tg(ori_pred, rest_pred)
            fid_var_list.append(fid_var)

        # import ipdb; ipdb.set_trace()
        fid_inv_best = array_best(fid_inv_list)
        fid_var_best = array_min(fid_var_list)
        sparsity = np.array(sparsity_list)

        return sparsity, fid_inv_list, fid_inv_best, fid_var_list, fid_var_best


def array_min(values):
    if len(values) == 0:
        return values
    best_values = [values[0], ]
    best = values[0]
    for i in range(1, len(values)):
        if best > values[i]:
            best = values[i]
        best_values.append(best)
    return np.array(best_values)


def array_best(values):
    if len(values) == 0:
        return values
    best_values = [values[0], ]
    best = values[0]
    for i in range(1, len(values)):
        if best < values[i]:
            best = values[i]
        best_values.append(best)
    return np.array(best_values)


