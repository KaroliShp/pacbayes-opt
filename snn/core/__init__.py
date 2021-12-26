# Retrieve the package location
import os
import snn
import inspect
package_path = os.path.dirname(inspect.getfile(snn))

import tensorflow as tf
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True