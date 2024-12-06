import os
import torch
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing as mp
from multiprocessing import Process

from ctmexplainer.xgraph.dataset.tg_dataset import load_tg_dataset, load_explain_idx
from ctmexplainer.xgraph.dataset.utils_dataset import construct_tgat_neighbor_finder

from ctmexplainer.xgraph.models.ext.tgat.module import TGAN
from ctmexplainer.xgraph.models.ext.tgn.model.tgn import TGN
from ctmexplainer.xgraph.models.ext.tgn.utils.data_processing import (
    compute_time_statistics,
)
from ctmexplainer import ROOT_DIR

def seed_everything(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def start_multi_process(explainer, target_event_idxs, parallel_degree):
    mp.set_start_method("spawn")
    process_list = []
    size = len(target_event_idxs) // parallel_degree
    split = [i * size for i in range(parallel_degree)] + [len(target_event_idxs)]
    return_dict = mp.Manager().dict()
    for i in range(parallel_degree):
        p = Process(
            target=explainer[i],
            kwargs={
                "event_idxs": target_event_idxs[split[i] : split[i + 1]],
                # "event_idxs": target_event_idxs[0],
                "return_dict": return_dict,
            },
        )
        process_list.append(p)
        p.start()


    for p in process_list:
        p.join()

    explain_results = [return_dict[event_idx] for event_idx in target_event_idxs]
    return explain_results


# def start_multi_process(explainer, target_event_idxs, parallel_degree):
#     mp.set_start_method('spawn')
#     return_dict = mp.Manager().dict()
#     pool = mp.Pool(parallel_degree)
#     for i, e_idx in enumerate(target_event_idxs):
#         pool.apply_async( partial(explainer[i%parallel_degree], event_idxs=[e_idx,], return_dict=return_dict, device=i%4) )

#     pool.close()
#     pool.join()

#     import ipdb; ipdb.set_trace()
#     explain_results = [return_dict[event_idx] for event_idx in target_event_idxs ]
#     return explain_results


@hydra.main(config_path="config", config_name="config")
def pipeline(config: DictConfig):
    # SEED
    seed_everything(config.seed)

    # model config
    config.models.param = config.models.param[config.datasets.dataset_name]
    config.models.ckpt_path = str(
        ROOT_DIR
        / "xgraph"
        / "models"
        / "checkpoints"
        / f"{config.models.model_name}_{config.datasets.dataset_name}_best.pth"
    )

    # dataset config
    config.datasets.dataset_path = str(
        ROOT_DIR / "xgraph" / "dataset" / "data" / f"{config.datasets.dataset_name}.csv"
    )
    config.datasets.explain_idx_filepath = str(
        ROOT_DIR
        / "xgraph"
        / "dataset"
        / "explain_index"
        / f"{config.datasets.explain_idx_filename}.csv"
    )

    # explainer config
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    config.explainers.results_dir = str(ROOT_DIR.parent / "benchmarks" / "results")
    config.explainers.mcts_saved_dir = str(ROOT_DIR / "xgraph" / "saved_mcts_results")
    config.explainers.cmomcts_saved_dir = str(ROOT_DIR / "xgraph" / "saved_cmomcts_results")
    config.explainers.explainer_ckpt_dir = str(ROOT_DIR / "xgraph" / "explainer_ckpts")

    print(OmegaConf.to_yaml(config))

    # import ipdb; ipdb.set_trace()

    if torch.cuda.is_available() and config.explainers.use_gpu:
        device = torch.device("cuda", index=config.device_id)
    else:
        device = torch.device("cpu")

    # DONE: only use tgat processed data
    events, edge_feats, node_feats = load_tg_dataset(config.datasets.dataset_name)
    target_event_idxs = load_explain_idx(config.datasets.explain_idx_filepath, start=0)

    ngh_finder = construct_tgat_neighbor_finder(events)

    if config.models.model_name == "tgat":
        model = TGAN(
            ngh_finder,
            node_feats,
            edge_feats,
            device=device,
            attn_mode=config.models.param.attn_mode,
            use_time=config.models.param.use_time,
            agg_method=config.models.param.agg_method,
            num_layers=config.models.param.num_layers,
            n_head=config.models.param.num_heads,
            num_neighbors=config.models.param.num_neighbors,
            drop_out=config.models.param.dropout,
        )
    elif config.models.model_name == "tgn":  # DONE: added tgn
        (
            mean_time_shift_src,
            std_time_shift_src,
            mean_time_shift_dst,
            std_time_shift_dst,
        ) = compute_time_statistics(events.u.values, events.i.values, events.ts.values)
        model = TGN(
            ngh_finder,
            node_feats,
            edge_feats,
            device=device,
            n_layers=config.models.param.num_layers,
            n_heads=config.models.param.num_heads,
            dropout=config.models.param.dropout,
            use_memory=True,  # True
            forbidden_memory_update=True,  # True
            memory_update_at_start=False,  # False
            message_dimension=config.models.param.message_dimension,
            memory_dimension=config.models.param.memory_dimension,
            embedding_module_type="graph_attention",  # fix
            message_function="identity",  # fix
            mean_time_shift_src=mean_time_shift_src,
            std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst,
            std_time_shift_dst=std_time_shift_dst,
            n_neighbors=config.models.param.num_neighbors,
            aggregator_type="last",  # fix
            memory_updater_type="gru",  # fix
            use_destination_embedding_in_message=False,
            use_source_embedding_in_message=False,
            dyrep=False,
        )
    else:
        raise NotImplementedError("Not supported.")

    # load model checkpoints
    state_dict = torch.load(config.models.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)


    from tgnnexplainer.xgraph.method.ctms_model_5 import CTMs

    explainer = CTMs(
        model,
        config.models.model_name,
        config.explainers.explainer_name,
        config.datasets.dataset_name,
        events,
        config.explainers.param.explanation_level,
        device=device,
        results_dir=config.explainers.results_dir,
        train_epochs=config.explainers.param.train_epochs,
        explainer_ckpt_dir=config.explainers.explainer_ckpt_dir,
        reg_coefs=config.explainers.param.reg_coefs,
        batch_size=config.explainers.param.batch_size,
        lr=config.explainers.param.lr,
        debug_mode=config.explainers.debug_mode,
        threshold_num=config.explainers.param.threshold_num

    # run the explainer
    #  =================================================if need test ======================
    import csv
    target_event_idxs = [item for i, item in enumerate(target_event_idxs) if i != 57 and int(item) != 140977]
    # target_event_idxs = [target_event_idxs[0],target_event_idxs[1]]
    # target_event_idxs = [target_event_idxs[0]]
    # print(target_event_idxs)
    start_time = time.time()
    explain_results = explainer(event_idxs=target_event_idxs)
    end_time = time.time()
    print(f"runtime: {end_time - start_time:.2f}s")
    runtime = end_time - start_time
    with open('runtime', 'a', newline='') as csvfile:  # 'a'模式表示追加模式
        writer = csv.writer(csvfile)
        writer.writerow([config.explainers.explainer_name, config.models.model_name, config.datasets.dataset_name,runtime])
    # exit(0)
    from tgnnexplainer.xgraph.evaluation.metrics_tg import EvaluatorCTMS

    evaluator = EvaluatorCTMS(
        model_name=config.models.model_name,
        explainer_name=config.explainers.explainer_name,
        dataset_name=config.datasets.dataset_name,
        explainer=explainer,
        results_dir=config.explainers.results_dir,
        threshold_num=config.explainers.param.threshold_num
    )  # DONE: updated

    if config.evaluate:
        evaluator.evaluate(explain_results, target_event_idxs)
    else:
        print("no evaluate.")
    # import ipdb; ipdb.set_trace()
    # exit(0)


if __name__ == "__main__":
    pipeline()
