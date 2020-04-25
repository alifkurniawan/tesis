"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

from preprocessing import process_raw_data

from models import *
from training import train_model
from util import contruct_dataloader_from_disk

data_paths = ['30', '50', '70', '90', '95', '100']
hidden_sizes = [32, 64, 128, 256, 512, 800]

def run_experiment(parser, use_gpu):
    # parse experiment specific command line arguments
    parser.add_argument('--learning-rate', dest='learning_rate', type=float,
                        default=0.001, help='Learning rate to use during training.')
    args, _unknown = parser.parse_known_args()

    # pre-process data
    process_raw_data(use_gpu, force_pre_processing_overwrite=False)

    for hidden_size in hidden_sizes:
        for path in data_paths:
            # run experiment
            training_file = "data/preprocessed/training_" + path + "'.hdf5"
            validation_file = "data/preprocessed/validation.hdf5"

            model = ExampleModel(21, hidden_size, args.minibatch_size, use_gpu=use_gpu)  # embed size = 21

            train_loader = contruct_dataloader_from_disk(training_file, args.minibatch_size)
            validation_loader = contruct_dataloader_from_disk(validation_file, args.minibatch_size)

            train_model_path = train_model(data_set_identifier="TRAIN-hidden" + str(hidden_size) + "-" + path + "_150",
                                           model=model,
                                           train_loader=train_loader,
                                           validation_loader=validation_loader,
                                           learning_rate=args.learning_rate,
                                           minibatch_size=args.minibatch_size,
                                           eval_interval=args.eval_interval,
                                           hide_ui=args.hide_ui,
                                           use_gpu=use_gpu,
                                           minimum_updates=args.minimum_updates)

            print(train_model_path)
