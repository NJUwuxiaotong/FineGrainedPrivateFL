"""Parser options."""

import argparse


def options():
    """Construct the central argument parser, filled with useful defaults."""
    parser = argparse.ArgumentParser(
        description='Reconstruct some image from a trained model in '
                    'federated learning environment.')

    # client information:
    parser.add_argument("--client_no", default=10, type=int,
                        help='the number of clients')
    parser.add_argument("--client_ratio", default=0.4, type=float,
                        help='the ratio of clients of global model update')
    parser.add_argument("--is_iid", default=True, type=bool,
                        help='whether the dataset is dispatched iid')
    parser.add_argument("--is_balanced", default=True, type=bool,
                        help="whether the dataset is balanced")
    parser.add_argument("--privacy_budget", default=2e4, type=float,
                        help="privacy budget of each client")
    parser.add_argument("--broken_probability", default=0.001, type=float,
                        help="broken probability for gaussian mechanism")
    parser.add_argument("--perturb_mechanism", default="ALG_rLapPGrad15",
                        type=str,
                        help="which perturbation mechanism executed by clients")
    parser.add_argument("--noise_dist", default="laplace", type=str,
                        help="which distribution the noise is drawn from")
    parser.add_argument("--clip_norm", default=4, type=float,
                        help="clipping norm applied to gaussian mechanism")
    parser.add_argument("--sigma", default=5e-4, type=float,
                        help="privacy level")
    parser.add_argument("--grad_ratio", default=0.999996, type=float,
                        help="the ratio of gradients are updated")

    # dataset information
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    # model training information
    parser.add_argument('--model_name', default='resnet34', type=str,
                        help='Vision model.')
    parser.add_argument("--round_no", default=400, type=int,
                        help='the number of round of global model')

    # training for clients
    parser.add_argument('--epoch_no', default=5, type=int,
                        help='If using a trained model, how many epochs was '
                             'it trained?')
    parser.add_argument('--batch_size', default=50, type=int,
                        help="the size at each batch")
    parser.add_argument('--lr', default=0.001, type=float,
                        help='the learning ratio in model training')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay')

    # invert gradient attack
    parser.add_argument('--num_images', default=3, type=int,
                        help='How many images should be recovered from the '
                             'given gradient.')
    parser.add_argument('--attack_app', default=None, type=str,
                        help='attack approach')
    parser.add_argument('--attack_no', default=1, type=int,
                        help='The number of attacks')
    parser.add_argument('--demo_target', default=False, type=bool,
                        help='Cifar validation image used for reconstruction.')
    parser.add_argument('--dtype', default='float', type=str,
                        help='Data type used during reconstruction '
                             '[Not during training!].')
    parser.add_argument('--trained_model', action='store_true',
                        help='Use a trained model.')
    parser.add_argument('--accumulation', default=2, type=int,
                        help='Accumulation 0 is rec. from gradient, '
                             'accumulation > 0 is reconstruction '
                             'from fed. averaging.')
    parser.add_argument('--label_flip', default=False, action='store_true',
                        help='Dishonest server permuting weights '
                             'in classification layer.')

    # Rec. parameters
    parser.add_argument('--optim', default='ours', type=str,
                        help='Use our reconstruction method or the DLG method.')
    parser.add_argument('--restarts', default=1, type=int,
                        help='How many restarts to run.')
    parser.add_argument('--cost_fn', default='sim', type=str,
                        help='Choice of cost function.')
    parser.add_argument('--indices', default='def', type=str,
                        help='Choice of indices from the parameter list.')
    parser.add_argument('--weights', default='equal', type=str,
                        help='Weigh the parameter list differently.')

    parser.add_argument('--optimizer', default='adam', type=str,
                        help='Weigh the parameter list differently.')
    parser.add_argument('--signed', action='store_false',
                        help='Do not used signed gradients.')
    parser.add_argument('--boxed', default=True, action='store_false',
                        help='Do not used box constraints.')

    parser.add_argument('--scoring_choice', default='loss', type=str,
                        help='How to find the best image between all restarts.')
    parser.add_argument('--init', default='randn', type=str,
                        help='Choice of image initialization.')
    parser.add_argument('--tv', default=1e-4, type=float,
                        help='Weight of TV penalty.')

    # Files and folders:
    parser.add_argument('--save_image', default=True, action='store_true',
                        help='Save the output to a file.')
    parser.add_argument('--image_path', default='attack_results/images/',
                        type=str, help='the path to save the reconstruction '
                                       'image')
    parser.add_argument('--model_path', default='models/', type=str)
    parser.add_argument('--table_path', default='tables/', type=str)
    parser.add_argument('--data_path', default='~/data', type=str)

    # Debugging:
    parser.add_argument('--name', default='iv', type=str,
                        help='Name tag for the result table and model.')
    parser.add_argument('--deterministic', action='store_true',
                        help='Disable CUDNN non-determinism.')
    parser.add_argument('--dryrun', action='store_true',
                        help='Run everything for just one step to '
                             'test functionality.')
    parser.add_argument('--soft_labels', action='store_true',
                        help='Do not use the provided label when using L-BFGS (This can stabilize it).')

    return parser

