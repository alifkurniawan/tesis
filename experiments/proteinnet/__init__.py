"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

from preprocessing import process_raw_data
from rgn import RGN
import time
from utgn import *
from training import train_model
from util import contruct_dataloader_from_disk

data_paths = ['30']


def run_experiment(parser, use_gpu):
    # parse experiment specific command line arguments
    parser.add_argument('--learning-rate', dest='learning_rate', type=float,
                        default=0.001, help='Learning rate to use during training.')
    args, _unknown = parser.parse_known_args()

    # pre-process data
    process_raw_data(use_gpu, force_pre_processing_overwrite=False)
    start_compute_grad = time.time()
    for path in data_paths:
        # run experiment
        training_file = "data/preprocessed/training_" + path + ".hdf5"
        validation_file = "data/preprocessed/validation.hdf5"

        if args.skenario is 1:
            model = RGN(embedding_size=42, use_gpu=use_gpu, minibatch_size=args.minibatch_size, pretraining=-1)
        elif args.skenario is 2:
            model = UTGN(embedding_size=42, use_gpu=use_gpu, batch_size=args.minibatch_size, pretraining=-1)
        elif args.skenario is 3:
            model = RGN(embedding_size=768 + 21, use_gpu=use_gpu, minibatch_size=args.minibatch_size, use_pssm=False, use_token=True)
        elif args.skenario is 4:
            model = UTGN(embedding_size=768 + 21, use_gpu=use_gpu, batch_size=args.minibatch_size, use_pssm=False, use_token=True)
        elif args.skenario is 5:
            model = RGN(embedding_size=21, use_gpu=use_gpu, minibatch_size=args.minibatch_size, pretraining=-1, use_pssm=False)
        elif args.skenario is 6:
            model = UTGN(embedding_size=21, use_gpu=use_gpu, batch_size=args.minibatch_size, pretraining=-1, use_pssm=False)
        elif args.skenario is 7:
            model = RGN(embedding_size=768, use_gpu=use_gpu, minibatch_size=args.minibatch_size, use_aa=False, use_pssm=False, use_token=True)
        elif args.skenario is 8:
            model = UTGN(embedding_size=768, use_gpu=use_gpu, batch_size=args.minibatch_size, use_aa=False, use_pssm=False, use_token=True)


        train_loader = contruct_dataloader_from_disk(training_file, args.minibatch_size)
        validation_loader = contruct_dataloader_from_disk(validation_file, args.minibatch_size)
        identifier = "skenario{0}".format(args.skenario)
        train_model_path = train_model(
            data_set_identifier=identifier,
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            learning_rate=args.learning_rate,
            minibatch_size=args.minibatch_size,
            eval_interval=args.eval_interval,
            hide_ui=args.hide_ui,
            use_gpu=use_gpu,
            optimizer_type=args.optimizer_type,
            restart=args.restart,
            minimum_updates=args.minimum_updates)

        print(train_model_path)
    end = time.time()
    print("Training time:", end - start_compute_grad)
