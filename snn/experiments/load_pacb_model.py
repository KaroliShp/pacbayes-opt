import os
import sys
sys.path.insert(0, "/snn/")
sys.path.insert(0, "/")

import tensorflow as tf

from snn.core import package_path
from snn.core.parse_args import CompleteParser, Interpreter
from snn.core.utils import deserialize
from snn.core.data_fn import load_binary_mnist
from snn.core.fc import FC


if __name__ == '__main__':
    seed = 100
    lr = 0.0001
    path = os.path.join(package_path, "experiments", "binary_mnist", ("model_mean_opt{}_LR{}_seed{}.pickle".format(True, lr, seed)))
    print("Loading model info")
    saved_model = deserialize(path)
    print("Model info loaded!")

    """
    print(type(saved_model))
    for k, v in saved_model.items():
        print(k)
    print(saved_model['PACBound'])
    print(saved_model['L2_PACB'])
    """
    """
    print(type(saved_model['PACB_weights']))
    print(len(saved_model['PACB_weights']))
    print(len(saved_model['PACB_weights'][-4:][0]))
    print(len(saved_model['PACB_weights'][-4:][1]))
    print(len(saved_model['PACB_weights'][-4:][2]))
    print(len(saved_model['PACB_weights'][-4:][3]))
    """
    """
    print(type(saved_model['log_prior_std']))
    print(type(saved_model['log_post_all']))
    print(len(saved_model['log_prior_std']))
    print(len(saved_model['log_post_all']))
    """

    corruption = 0.25
    layers = [600]
    scopes_list = ["hidden" + str(i+1) for i in range(len(layers))]
    scopes_list.append("output")
    print("Random labels corruption: " + str(corruption))
    (trainX, trainY), test_set = load_binary_mnist(corrupt_prob=corruption)
    model = FC(trainX, trainY, layers=[784] + layers + [1], scopes_list=scopes_list, graph=tf.Graph(),
                seed=100, initial_weights=saved_model['PACB_weights'][-1])
    """
    weights_rand_init = model.optimize(epochs=0)
    model.print_full_accuracy(*test_set)
    """
    og_model_path = os.path.join(package_path, "experiments", "binary_mnist", "FC_layers[600]_epochs120_seed100.pickle")
    _, weights_rand_init = deserialize(og_model_path)

    save_dict = {"log_post_all": True, "PACB_weights": True, "L2_PACB": False, "diff": False, "iter": 500*50,
                 "w*": model.get_model_weights(), "mean_weights": True, "var_weights": True, "PACBound": True,
                 "B_val": True, "KL_val": True, "test_acc": True, "train_acc": True, "log_prior_std": True}
    # Optimize the pac bayes bound with this newly trained prior
    model.PACB_init
    # Checkpoint optimization runs periodically (absolute),
    model.optimize_PACB(weights_rand_init, 0, learning_rate=0.0001, drop_lr=10, lr_factor=1,
                        save_dict=save_dict, trainWeights=False)
    model.log_prior_std = saved_model['log_prior_std'][-1]
    model.log_post_all = saved_model['log_post_all'][-1]
    model.evaluate_SNN_accuracy(test_set[0], test_set[1], weights_rand_init, N_SNN_samples=1000, save_dict=save_dict)