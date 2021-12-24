import os
import sys
sys.path.insert(0, "/snn/")
sys.path.insert(0, "/")

from snn.core import package_path
from snn.core.parse_args import CompleteParser, Interpreter
from snn.core.utils import deserialize


if __name__ == '__main__':
    complete_args = CompleteParser().parse()
    _, _, save_path = Interpreter(complete_args).interpret()
    deserialization_path = os.path.join(package_path, "experiments", save_path)
    print("Loading model weights saved in ", deserialization_path)
    model_weights, weights_rand_init = deserialize(deserialization_path)
    print("Model weights loaded!")
    model, test_set, _ = Interpreter(complete_args).interpret(model_weights)
    weights_rand_init = model.optimize(epochs=0)
    model.print_full_accuracy(*test_set)