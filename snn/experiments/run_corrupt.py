import os
import sys
sys.path.insert(0, "/snn/")
sys.path.insert(0, "/")

from snn.core import package_path
from snn.core.parse_args import BasicParser, CompleteParser, Interpreter
from snn.core.utils import serialize, deserialize
import copy


def run_sgd(model, epochs):
    """
        Runs SGD for a predefined number of epochs and saves the resulting model.
    """
    print("Training full network")
    weights_rand_init = model.optimize(epochs=epochs)
    # weights_rand_init = model.optimize(epochs=epochs, batch_size=55000, learning_rate=0.03)
    print("Model optimized!!!")

    return [model.get_model_weights(), weights_rand_init]


def sgd_main(basic_args):
    print("\n--SGD--\n")
    model, test_set, save_path, train_set = Interpreter(basic_args).interpret()
    model.test_X = test_set[0]
    model.test_Y = test_set[1]
    saved_entities = run_sgd(model, basic_args["sgd_epochs"])
    model.print_full_accuracy(*test_set)
    serialization_path = os.path.join(package_path, "experiments", save_path)
    print("Saving run in ", serialization_path)
    serialize(saved_entities, serialization_path, basic_args["overwrite"])
    print("SGD run complete!")
    print("\n--SGD--\n")
    return train_set


def run_pacb(weights_rand_init, model, test_set, epochs, learning_rate, drop_lr, lr_factor, seed, trainw):
    testX, testY = test_set

    save_dict = {"log_post_all": True, "PACB_weights": True, "L2_PACB": False, "diff": False, "iter": 500*50,
                 "w*": model.get_model_weights(), "mean_weights": True, "var_weights": True, "PACBound": True,
                 "B_val": True, "KL_val": True, "test_acc": True, "train_acc": True, "log_prior_std": True}
    # Optimize the pac bayes bound with this newly trained prior
    model.PACB_init
    # Checkpoint optimization runs periodically (absolute),
    model.optimize_PACB(weights_rand_init, epochs, learning_rate=learning_rate, drop_lr=drop_lr, lr_factor=lr_factor,
                        save_dict=save_dict, trainWeights=trainw)
    model.evaluate_SNN_accuracy(testX, testY, weights_rand_init, N_SNN_samples=50, save_dict=save_dict)

    path = os.path.join(package_path, "experiments", "binary_mnist",
                        ("model_mean_opt{}_LR{}_seed{}.pickle".format(trainw, learning_rate, seed)))
    model.save_output(path=path)

    path_log = os.path.join(package_path, "experiments", "binary_mnist",
                        ("model_mean_opt{}_LR{}_seed{}.csv".format(trainw, learning_rate, seed)))
    model.save_logging_info(path_log)


def pacb_main(train_set, complete_args):
    print("\n--PAC BAYES--\n")
    _, _, save_path, _ = Interpreter(complete_args).interpret()
    deserialization_path = os.path.join(package_path, "experiments", save_path)
    print("Loading model weights saved in ", deserialization_path)
    model_weights, weights_rand_init = deserialize(deserialization_path)
    print("Model weights loaded!")
    model, test_set, _, train_test = Interpreter(complete_args).interpret(model_weights)

    print("OVERWRITING MODEL TRAIN/TEST SETS")
    model.X = train_set[0]
    model.Y = train_set[1]
    print("TRAIN DATA IDENTICAL X: " + str((model.X == train_test[0]).all()))
    print("TRAIN DATA IDENTICAL Y: " + str((model.Y == train_test[1]).all()))

    run_pacb(weights_rand_init, model, test_set, complete_args["pacb_epochs"], complete_args["lr"],
             complete_args["drop_lr"], complete_args["lr_factor"], complete_args["seed"], complete_args["trainw"])
    print("PAC-Bayes run complete!")


if __name__ == '__main__':
    complete_args = CompleteParser().parse()
    train_set = sgd_main(complete_args)
    pacb_main(train_set, complete_args)