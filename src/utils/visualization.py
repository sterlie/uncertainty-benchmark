import torch
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.pylabtools import figsize
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def entropy(id: torch.tensor, ood: dict, kde=True):
    res = pd.DataFrame()
    if id is not None:
        for k_ in ['total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty']:
            res = pd.concat([res, pd.DataFrame({'type': ['ID'] * len(id[k_]),
                                        'uncertainty': [k_] * len(id[k_]),
                                        'Entropy': list(
                                            id[k_])})],
                    ignore_index=True, )
    for k, v in ood.items():
        for k_ in ['total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty']:
            res = pd.concat(
                [res, pd.DataFrame({'type': [k] * len(v[k_]), 'Entropy': list(v[k_]), 'uncertainty': [k_] * len(v[k_])})],
                ignore_index=True, )
    g = sns.FacetGrid(res, col="uncertainty", hue="type", sharex=False)
    g.map(sns.histplot, "Entropy", stat="probability", element="step", kde=kde)
    g.add_legend()
    return g

def roc(id_scores, ood_scores):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    for k in id_scores.keys():
        for i, k_ in enumerate(['total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty']):
            true_labels = np.concatenate([np.zeros_like(id_scores[k][k_]), np.ones_like(ood_scores[k][k_])])
            uncertainty_scores = np.concatenate([id_scores[k][k_], ood_scores[k][k_]])  # higher = more OOD-like
            fpr, tpr, _ = roc_curve(true_labels, uncertainty_scores)
            auroc = roc_auc_score(true_labels, uncertainty_scores)
            axs[i].plot(fpr, tpr, label=f"{k}_AUROC={auroc:.3f}")
            axs[i].plot([0,1],[0,1],'--', color='gray')
    plt.legend()

    return plt

def error_rate(uncertainty_scores, predicted_labels, true_labels, kde=True):
    res = pd.DataFrame()
    # predicted_labels = torch.argmax(predictions, dim=-1).numpy()
    for k_ in ['total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty']:
        print(predicted_labels.shape)
        uncertainties = uncertainty_scores[k_]
        print(uncertainties.shape)
        errors = (predicted_labels != true_labels)
        res = pd.concat([res, pd.DataFrame({'Error Rate':errors, 'Uncertainty Score': uncertainties, 'uncertainty': [k_]*len(uncertainties)})], ignore_index=True)

    # Create FacetGrid
    g = sns.FacetGrid(res, col="uncertainty", sharex=False, sharey=False)

    # Plot 2D histogram
    g.map_dataframe(sns.histplot, x="Uncertainty Score", y="Error Rate", stat="probability", bins=100)

    # Overlay mean line per facet
    def mean_line(data, **kwargs):
        # Bin x values
        bins = np.linspace(data["Uncertainty Score"].min(), data["Uncertainty Score"].max(), 30)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        means = [data.loc[(data["Uncertainty Score"] >= bins[i]) & (
                    data["Uncertainty Score"] < bins[i + 1]), "Error Rate"].mean()
                 for i in range(len(bins) - 1)]
        plt.plot(bin_centers, means, color='red', linewidth=2)

    g.map_dataframe(mean_line)
    g.add_legend()

    return g

def mean_error_rate(uncertainty_scores, mean_predictions, true_labels, kde=True):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for i, k_ in enumerate(['total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty']):
        res = pd.DataFrame()
        for k in uncertainty_scores.keys():
            predicted_labels = torch.argmax(mean_predictions[k], dim=-1).numpy()
            scores = uncertainty_scores[k][k_]
            df = pd.DataFrame({'Uncertainty Score': scores, 'Error Rate':predicted_labels!=true_labels})

            bins = np.linspace(df["Uncertainty Score"].min(), df["Uncertainty Score"].max(), 50)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            means = [df.loc[(df["Uncertainty Score"] >= bins[i]) & (
                    df["Uncertainty Score"] < bins[i + 1]), "Error Rate"].mean()
                     for i in range(len(bins) - 1)]

            res = pd.concat([res, pd.DataFrame({'Uncertainty Score': bin_centers, 'Error Rate': means,'type':[k]*len(bin_centers), 'uncertainty': [k_]*len(bin_centers)})], ignore_index=True)

        sns.lineplot(res, x='Uncertainty Score', y='Error Rate', hue='type', ax=axs[i])
        axs[i].set_title(f"{k_}")
    return plt

def trend(uncertainty_scores):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    for i, k_ in enumerate(['total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty']):
        rows = []
        for name in uncertainty_scores.keys():
            scores = uncertainty_scores[name][k_].detach().cpu().numpy() if isinstance(uncertainty_scores[name][k_], torch.Tensor) else uncertainty_scores[name][k_]
            rows.append({
                'Uncertainty Score': scores,
                'distortion_cat': name,
                'type': k_
            })
        res = pd.DataFrame(rows).explode('Uncertainty Score').reset_index(drop=True)
        res["distortion"] = pd.Categorical(res["distortion_cat"], ordered=True).codes
        # sns.regplot(data=res, x="distortion", y="Uncertainty Score", label=k_, x_estimator=np.mean, order=1, ax=ax[i])

        sns.lineplot(x="distortion", y="Uncertainty Score", hue='type', data=res, ax=ax[i])
        ax[i].set_xticks(res["distortion"])
        ax[i].set_xticklabels(res["distortion_cat"])
        ax[i].legend(loc='best')
        print("figure done")
    return plt


def compare_uncertainties(uncertainties, feature):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for i, k_ in enumerate(['total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty']):
        res = pd.DataFrame()
        for name in uncertainties.keys():
            res = pd.concat([res, pd.DataFrame({'Uncertainty Score':uncertainties[name][k_], f'{feature}': [f'{name}']*len(uncertainties[name][k_])})])

        print(res)
        sns.barplot(data=res, x=feature, y='Uncertainty Score', estimator='mean', errorbar='sd', ax=axs[i])
        axs[i].set_title(f"{k_}")

    return plt

def aleatoric_score_induced_uncertainty(uncertainties_per_method):
    uncertainty_type='aleatoric_uncertainty'
    fig, ax = plt.subplots()
    rows = []
    for method, uncertainties in uncertainties_per_method.items():
        for level, values in uncertainties.items():
            rows.append({
                'Uncertainty Level': level,
                'Uncertainty Score': values[uncertainty_type],
                'Method': method
            })
    res = pd.DataFrame(rows).explode('Uncertainty Score').reset_index(drop=True)
    sns.barplot(data=res, x='Uncertainty Level', y='Uncertainty Score', hue='Method', estimator='mean', errorbar='sd', ax=ax)

    return plt

def aleatoric_trend(uncertainties_per_method, uncertainty_type = 'aleatoric_uncertainty'):
    fig, ax = plt.subplots()
    rows = []
    for method, uncertainties in uncertainties_per_method.items():
        for level, values in uncertainties.items():
            rows.append({
                'Uncertainty Level': level,
                'Uncertainty Score': values[uncertainty_type].detach().cpu().numpy() if isinstance(values[uncertainty_type], torch.Tensor) else values[uncertainty_type],
                'Method': method
            })
    res = pd.DataFrame(rows).explode('Uncertainty Score').reset_index(drop=True)
    res["UncertaintyLevelNum"] = pd.Categorical(res["Uncertainty Level"], ordered=True).codes

    res.sort_values('UncertaintyLevelNum')
    g = sns.relplot(data=res, x="Uncertainty Level", y="Uncertainty Score", hue='Method', estimator='mean', kind='line', dashes=False, markers=True)

    g.set_xticklabels(rotation=60, ha="right")
    g.tight_layout()
    return g

def moving_out_of_distribution(uncertainty_per_distortion, uncertainty_type="total_uncertainty"):
    '''
    plot epistemic uncertainty changes based on different level of distortions
    (how uncertainty changes when the data moves out of distribution)
    :param uncertainty_per_distortion:
    :return:
    '''
    distortion_types = uncertainty_per_distortion.keys()
    distortion_nums = len(distortion_types)
    n_rows = int(np.sqrt(distortion_nums))
    n_cols = int(np.ceil(distortion_nums/int(np.sqrt(distortion_nums))))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*7, n_rows*5))
    axs = np.atleast_1d(axs)
    for i, distortion in enumerate(distortion_types):
        rows = []
        uncertainties = uncertainty_per_distortion[distortion]
        for distortion_level, uncertainty in uncertainties.items():
            rows.append({
                'Distortion Level': distortion_level,
                'Uncertainty Score': uncertainty[uncertainty_type].detach().cpu().numpy() if isinstance(uncertainty[uncertainty_type], torch.Tensor) else uncertainty[uncertainty_type],
            })

        res = pd.DataFrame(rows).explode('Uncertainty Score').reset_index(drop=True)
        sns.lineplot(x="Distortion Level", y="Uncertainty Score", data=res, ax=axs[i//n_cols, i%n_cols])
        axs[i//n_cols, i%n_cols].tick_params(axis='x', rotation=60)
    plt.tight_layout()

    return plt

def moving_out_of_distribution_compare(uncertainty_per_distortion, legend_map, palette, uncertainty_type="epistemic_uncertainty"):
    '''
    plot epistemic uncertainty changes based on different level of distortions
    (how uncertainty changes when the data moves out of distribution)
    :param uncertainty_per_distortion:
    :return:
    '''
    distortion_types = uncertainty_per_distortion.keys()
    distortion_nums = len(distortion_types)
    n_rows = int(np.sqrt(distortion_nums))
    n_cols = int(np.ceil(distortion_nums/int(np.sqrt(distortion_nums))))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*7, n_rows*5))
    axs = np.atleast_2d(axs)
    sns.set_style("white")
    for i, distortion in enumerate(distortion_types):
        rows = []
        uncertainties = uncertainty_per_distortion[distortion]
        for m in uncertainties.keys():
            for distortion_level, uncertainty in uncertainties[m].items():
                df = pd.DataFrame({'gt': uncertainty['ground_truth'], 'score': uncertainty[uncertainty_type]})
                undertainty_scores = (uncertainty[uncertainty_type] - uncertainty[uncertainty_type].min()) / (uncertainty[uncertainty_type].max() - uncertainty[uncertainty_type].min())
                rows.append({
                    'Distortion Level': distortion_level,
                    'Uncertainty Score': uncertainty[uncertainty_type].detach().cpu().numpy() if isinstance(uncertainty[uncertainty_type], torch.Tensor) else uncertainty[uncertainty_type], #undertainty_scores, #uncertainty[uncertainty_type].detach().cpu().numpy() if isinstance(uncertainty[uncertainty_type], torch.Tensor) else uncertainty[uncertainty_type], ##df[df['gt']==5]['score'].values, #df.groupby('gt').mean()['score'].values, #
                    'method': legend_map[m],
                })

        res = pd.DataFrame(rows).explode('Uncertainty Score').reset_index(drop=True)
        sns.lineplot(x="Distortion Level", y="Uncertainty Score", hue='method', data=res, ax=axs[i//n_cols, i%n_cols], palette=palette)
        # axs[i//n_cols, i%n_cols].tick_params(axis='x', rotation=60)
        axs[i // n_cols, i % n_cols].set_title(distortion)
    plt.tight_layout()


    return plt


def ood_roc(uncertainty_per_distortion, id_uncertainty, uncertainty_type = 'epistemic_uncertainty'):
    distortion_types = uncertainty_per_distortion.keys()
    distortion_nums = len(distortion_types)
    n_rows = int(np.sqrt(distortion_nums))
    n_cols = distortion_nums // int(np.sqrt(distortion_nums))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*7, n_rows*5))
    axs = np.atleast_1d(axs)
    for i, distortion in enumerate(distortion_types):
        rows = []
        uncertainties = uncertainty_per_distortion[distortion]
        for distortion_level, uncertainty in uncertainties.items():
            # predictions = uncertainty["predictions"].detach().cpu().numpy() if isinstance(uncertainty["predictions"],torch.Tensor) else uncertainty["predictions"]
            # predictions = np.argmax(predictions, axis=1)
            # gt = uncertainty["groundtruth"].detach().cpu().numpy() if isinstance(uncertainty["groundtruth"],torch.Tensor) else uncertainty["groundtruth"]
            # auroc = roc_auc_score(gt, predictions)
            # rows.append({
            #     'Distortion Level': distortion_level,
            #     'AUROC': auroc,
            #     'Task': 'Digit classification'
            # })

            uncertainty = uncertainty[uncertainty_type].detach().cpu().numpy() if isinstance(uncertainty[uncertainty_type],
                                                                               torch.Tensor) else uncertainty[
                uncertainty_type]
            true_labels = np.concatenate([np.zeros_like(id_uncertainty[uncertainty_type]), np.ones_like(uncertainty)])
            uncertainty_scores = np.concatenate([id_uncertainty[uncertainty_type], uncertainty])  # higher = more OOD-like
            auroc = roc_auc_score(true_labels, uncertainty_scores)
            rows.append({
                'Distortion Level': distortion_level,
                'AUROC': auroc,
                'Task': 'OOD classification'
            })

        res = pd.DataFrame(rows)
        sns.barplot(res, x='Distortion Level', y='AUROC', hue='Task', ax=axs[i])
        axs[i].tick_params(axis='x', rotation=60)
    plt.tight_layout()

    return plt
