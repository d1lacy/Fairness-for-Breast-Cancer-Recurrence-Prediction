import numpy as np
import os
from experiments.generate_plots import SupervisedPlotGenerator
from seldonian.utils.io_utils import load_pickle
from sklearn.metrics import log_loss,accuracy_score
import sys


# Get constraint name and epsilon as inputs
constraint_name = ""
while constraint_name not in ["overall_accuracy_equality", "equal_opportunity"]:
    constraint_name = input("\nEnter constraint name\n(overall_accuracy_equality, equal_opportunity, or false_negative_safety): ")

epsilon = -1
while epsilon not in [0.2, 0.1, 0.05]:
    epsilon = float(input("\nEnter an epsilon value\n(0.2, 0.1, 0.05): "))



specfile = f'specs/{constraint_name}_{epsilon}.pkl'
spec = load_pickle(specfile)

performance_metric = 'accuracy'
n_trials = 50
data_fracs = np.logspace(-3, 0, 15)
n_workers = 8
verbose = False
results_dir = f'results/{constraint_name}_{epsilon}'
os.makedirs(results_dir, exist_ok=True)
images_dir = f'images/'
os.makedirs(images_dir, exist_ok=True)
plot_savename = os.path.join(images_dir, f'{constraint_name}_{epsilon}.png')

dataset = spec.dataset
test_features = dataset.features
test_labels = dataset.labels

def perf_eval_fn(y_pred,y,**kwargs):
    if performance_metric == 'log_loss':
        return log_loss(y,y_pred)
    elif performance_metric == 'accuracy':
        v = np.where(y!=1.0,1.0-y_pred,y_pred)
        return sum(v)/len(v)
    

perf_eval_kwargs = {
    'X':test_features,
    'y':test_labels
}

plot_generator = SupervisedPlotGenerator(
    spec=spec,
    n_trials=n_trials,
    data_fracs=data_fracs,
    n_workers=n_workers,
    datagen_method='resample',
    perf_eval_fn=perf_eval_fn,
    constraint_eval_fns=[],
    results_dir=results_dir,
    perf_eval_kwargs=perf_eval_kwargs,
    )

plot_generator.run_baseline_experiment(
    model_name='random_classifier',verbose=verbose)

plot_generator.run_baseline_experiment(
    model_name='logistic_regression',verbose=verbose)

plot_generator.run_seldonian_experiment(verbose=verbose)

plot_generator.make_plots(fontsize=12,legend_fontsize=8, 
                            performance_label=performance_metric,
                            save_format='png',
                            savename=plot_savename,
    )