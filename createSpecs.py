import autograd.numpy as np
import os
from seldonian.parse_tree.parse_tree import (ParseTree,
    make_parse_trees_from_constraints)

from seldonian.dataset import DataSetLoader
from seldonian.utils.io_utils import (load_json,save_pickle)
from seldonian.spec import SupervisedSpec
from seldonian.models.models import (
    BinaryLogisticRegressionModel as LogisticRegressionModel) 
from seldonian.models import objectives

if __name__ == '__main__':
    data_pth = "./data/BC_Data_Proc.csv"
    metadata_pth = "./data/metadata_breast_cancer.json"
    save_dir = './specs'
    os.makedirs(save_dir,exist_ok=True)
    
    # Create dataset from data and metadata file
    regime='supervised_learning'
    sub_regime='classification'

    loader = DataSetLoader(regime=regime)

    dataset = loader.load_supervised_dataset(data_pth, metadata_pth)
    
    sensitive_col_names = dataset.meta_information['sensitive_col_names']

    # Use logistic regression model
    model = LogisticRegressionModel()
    
    # Set the primary objective to be log loss
    primary_objective = objectives.binary_logistic_loss
    

    # Define fairness constraints
    constraint_names = ["overall_accuracy_equality", "equal_opportunity"]
    epsilons = [0.2, 0.1, 0.05]
    deltas = [0.05]

    for constraint_name in constraint_names:
        for epsilon in epsilons:

            if constraint_name == "overall_accuracy_equality":
                constraint_strs = [f'min((ACC | [premenopause])/(ACC | [menopause]),(ACC | [menopause])/(ACC | [premenopause])) >= {1-epsilon}']
            
            elif constraint_name == "equal_opportunity":
                constraint_strs = [f'min((FNR | [premenopause])/(FNR | [menopause]),(FNR | [menopause])/(FNR | [premenopause])) >= {1-epsilon}']
    
            # For each constraint (in this case only one), make a parse tree
            parse_trees = make_parse_trees_from_constraints(
                constraint_strs,deltas,regime=regime,
                sub_regime=sub_regime,columns=sensitive_col_names)

            # Save spec object, using defaults where necessary
            spec = SupervisedSpec(
                dataset=dataset,
                model=model,
                parse_trees=parse_trees,
                sub_regime=sub_regime,
                frac_data_in_safety=0.5,
                primary_objective=primary_objective,
                initial_solution_fn=model.fit,
                use_builtin_primary_gradient_fn=True,
                optimization_technique='gradient_descent',
                optimizer='adam',
                optimization_hyperparams={
                    'lambda_init'   : np.array([0.5]),
                    'alpha_theta'   : 0.01,
                    'alpha_lamb'    : 0.01,
                    'beta_velocity' : 0.9,
                    'beta_rmsprop'  : 0.95,
                    'use_batches'   : False,
                    'num_iters'     : 1500,
                    'gradient_library': "autograd",
                    'hyper_search'  : None,
                    'verbose'       : True,
                }
            )

            spec_save_name = os.path.join(save_dir,f'{constraint_name}_{epsilon}.pkl')
            save_pickle(spec_save_name,spec)
            print(f"Saved Spec object to: {spec_save_name}")