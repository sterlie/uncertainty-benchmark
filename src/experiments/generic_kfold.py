import pickle

import numpy as np
import pandas as pd
import torch

from methods.method_factory import MethodFactory
from methods.test_time_augmentation import TTA
from load_data import *

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import os
from utils.visualization import *
from utils.id_ood_classification import *
import matplotlib.pyplot as plt


# train_dl, val_dl, test_dl = load_isic_data('/home/msafa/PhD/isic/MILK')
# print("data has been loaded,", test_dl.keys())

def main(config):
    # Print the configuration
    print(config)
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(config))

    ood_uncertainties = {}
    id_uncertainties = []
    for fold, train_dl, val_dl, test_dl in load_isic_data('../isic/MILK'):
        retrain_base_model = False
        method = MethodFactory.create(config)

        os.makedirs(os.path.join(config.output.path, config.method.name), exist_ok=True)
        os.makedirs(os.path.join(config.output.path, config.method.name, 'model'), exist_ok=True)
        os.makedirs(os.path.join(config.output.path, config.method.name, 'result'), exist_ok=True)
        os.makedirs(os.path.join(config.output.path, config.method.name, 'plots'), exist_ok=True)

        if not retrain_base_model and os.path.exists(
                os.path.join(config.output.base_model_path, f"base_model_{config.model.name}_{fold}.pt")):
            method.build_base_model(retrain=False, pretrained=os.path.join(config.output.base_model_path,
                                                                           f"base_model_{config.model.name}_{fold}.pt"),
                                    train_loader=train_dl, val_loader=val_dl)
        else:
            method.build_base_model(retrain=True, train_loader=train_dl, val_loader=val_dl,
                                    model_name=f"base_model_{config.model.name}_{fold}.pt")
        os.makedirs(os.path.join(config.output.path, config.method.name, 'model'), exist_ok=True)
        if os.path.exists(os.path.join(config.output.path, config.method.name, 'model', f"model_{fold}.pt")):
            method.build_method(rebuild=False, train_loader=train_dl,
                                pretrained=os.path.join(config.output.path, config.method.name, 'model',
                                                        f"model_{fold}.pt"))
        else:
            method.build_method(rebuild=True, train_loader=train_dl, valid_loader=val_dl,
                                model_name=f"model_{fold}.pt")

        #
        if os.path.exists(os.path.join(config.output.path, config.method.name, 'result', f"id_uncertainty_{fold}.pkl")):
            with open(os.path.join(config.output.path, config.method.name, 'result', f"id_uncertainty_{fold}.pkl"), 'rb') as f:
                id_uncertainty = pickle.load(f)
        else:
            id_uncertainty = method.measure_uncertainty(val_dl)
            with open(os.path.join(config.output.path, config.method.name, 'result', f"id_uncertainty_{fold}.pkl"), 'wb') as f:
                pickle.dump(id_uncertainty, f)
        id_uncertainties.append(id_uncertainty)

        for k, v in test_dl.items():
            for k_, v_ in v.items():
                if os.path.exists(os.path.join(config.output.path, config.method.name, 'result', f'testset_uncertainty_{k}_{k_}_{fold}.pkl')):
                    with open(os.path.join(config.output.path, config.method.name, 'result', f'testset_uncertainty_{k}_{k_}_{fold}.pkl'), 'rb') as f:
                        uncertainties = pickle.load(f)
                else:
                    uncertainties = method.measure_uncertainty(v_)
                    with open(os.path.join(config.output.path, config.method.name, 'result', f'testset_uncertainty_{k}_{k_}_{fold}.pkl'), 'wb') as f:
                        pickle.dump(uncertainties, f)

                if k in ood_uncertainties:
                    if k_ in ood_uncertainties[k]:
                        ood_uncertainties[k][k_].append(uncertainties)
                    else:
                        ood_uncertainties[k][k_] = [uncertainties]
                else:
                    ood_uncertainties[k]={}
                    ood_uncertainties[k][k_] = [uncertainties]


    for k in ood_uncertainties.keys():
        for k_ in ood_uncertainties[k].keys():
            ood_uncertainties[k][k_] = {
                key: np.concatenate([d[key].detach().cpu().numpy() for d in ood_uncertainties[k][k_]], axis=0)
                for key in ood_uncertainties[k][k_][0]
            }

    id_uncertainties = {
                key: np.concatenate([d[key].detach().cpu().numpy() for d in id_uncertainties], axis=0)
                for key in id_uncertainties[0]
            }


    plot =moving_out_of_distribution(ood_uncertainties)
    plot.savefig(os.path.join(config.output.path, config.method.name, 'plots', f"moving_out_of_dist.png"))
    plt.close()

    return ood_uncertainties

if __name__ == "__main__":
    ood_uncertainties = {}
    for method_name in ["mc_dropout", "ensemble", "swag", "LA", "ddu", "TTA", "het_xl", "entropy"]:
        with initialize(config_path="../configs", version_base=None):
            cfg = compose(config_name="config", overrides=[f"method={method_name}"])
        res = main(cfg)
        for k in res.keys():
            if k not in ood_uncertainties:
                ood_uncertainties[k] = {}
            ood_uncertainties[k][method_name] = res[k]


    os.makedirs(os.path.join(cfg.output.path, 'common', 'plots'), exist_ok=True)
    for type in ["total_uncertainty", "epistemic_uncertainty", "aleatoric_uncertainty", "mutual_information", "variance_total_uncertainty", "variance_epistemic_uncertainty", "variance_aleatoric_uncertainty"]:
        plot = moving_out_of_distribution_compare(ood_uncertainties, uncertainty_type=type)
        plot.savefig(os.path.join(cfg.output.path, 'common', 'plots', f'moving_out_of_dist_{type}.png'))
    plt.close()