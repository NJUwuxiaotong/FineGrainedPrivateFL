import collections
import datetime
import json
import numpy as np
import os
import time
import torch
import torchvision
from loss import Classification

from copy import deepcopy
from PIL import Image
from torch import nn

import utils
from attack.reconstruction_algorithms \
    import FedAvgReconstructor, GradientReconstructor
from attack import metrics
from attack.modules import MetaMonkey
from constant import consts
from federated_learning.models.mlp2layer import MLP2Layer
from federated_learning.models.cnn2layer import CNN2Layer
from federated_learning.models.cnn4layer import CNN4Layer

from federated_learning.nn.models import ResNet

torch.backends.cudnn.benchmark =  consts.BENCHMARK


start_time = time.time()


class FedAvgServer(object):
    def __init__(self, sys_args, sys_defs, sys_setup, valid_loader, valid_info,
                 class_no):
        # get arguments
        self.sys_args = sys_args
        self.sys_setup = sys_setup

        # client info
        self.client_no = sys_args.client_no
        self.client_ratio = sys_args.client_ratio

        # dataset and the corresponding dm and ds ([[[*]]]) from consts
        self.dataset = sys_args.dataset
        self.dm = torch.as_tensor(
            getattr(consts,
                    f"{self.dataset.upper()}_MEAN"), **sys_setup)[:, None, None]
        self.ds = torch.as_tensor(
            getattr(consts,
                    f"{self.dataset.upper()}_STD"), **sys_setup)[:, None, None]

        # example info
        self.example_shape = valid_info.example_shape
        self.example_channel = self.example_shape[0]
        self.example_row_pixel = self.example_shape[1]
        self.example_column_pixel = self.example_shape[2]
        self.class_no = class_no

        # test examples
        self.valid_loader = valid_loader
        self.valid_info = valid_info
        self.test_examples = None
        self.test_labels = None
        self.test_example_no = valid_info.example_no
        self.get_test_examples()

        # model info
        self.model_type = sys_args.model_name
        self.epoch_no = sys_args.epoch_no
        self.round_no = sys_args.round_no
        self.lr = sys_args.lr
        self.batch_size = sys_args.batch_size

        self.global_model = None
        self.model_shape = None
        self.center_radius_stats = None
        self.loss_fn = None
        self.current_client_model_params = None

        # attack information
        self.attack_no = sys_args.attack_no
        self.attack_rounds = list()
        self.attack_targets = list()
        self.attack_image_no = sys_args.num_images

        self.softmax = nn.Softmax(dim=1)

    def prepare_before_training(self):
        # prepare for the model training
        self.construct_model()
        self.global_model.to(**self.sys_setup)
        self.get_model_shape()
        self.get_center_radius_of_model()
        # prepare for the attack
        self.select_attack_rounds()

    def get_test_examples(self):
        examples = list()
        labels = list()
        for example, label in self.valid_loader:
            examples.extend(example.tolist())
            labels.extend(label.tolist())

        self.test_examples = \
            torch.tensor(examples, device=self.sys_setup["device"])
        self.test_labels = torch.tensor(labels, device=self.sys_setup["device"])

    def construct_model(self):
        if self.model_type == consts.MNIST_MLP_MODEL:
            num_neurons = [200, 200]
            self.global_model = MLP2Layer(self.example_shape,
                                          self.class_no,
                                          num_neurons)
            self.global_model.construct_model()
        elif self.model_type == consts.MNIST_CNN_MODEL:
            model_params = \
                {"conv1": {"in_channel": 1,
                           "out_channels": 32,
                           "kernel_size": 5,
                           "stride": 1,
                           "padding": 2},
                 "pool1": {"kernel_size": 2,
                           "stride": 2},
                 "conv2": {"in_channel": 32,
                           "out_channels": 64,
                           "kernel_size": 5,
                           "stride": 1,
                           "padding": 2},
                 "pool2": {"kernel_size": 2,
                           "stride": 2},
                 "fc": {"in_neuron": 7*7*64,
                        "out_neuron": 512}}
            self.global_model = CNN2Layer(
                self.example_shape,
                self.class_no, **model_params)
            self.global_model.initial_layers()
        elif self.model_type == consts.CIFAR10_CNN_MODEL:
            model_params = {"conv1": {"in_channel": 3,
                           "out_channels": 32,
                           "kernel_size": 3,
                           "stride": 1,
                           "padding": 1},
                 "pool1": {"kernel_size": 2,
                           "stride": 2},
                 "conv2": {"in_channel": 32,
                           "out_channels": 64,
                           "kernel_size": 3,
                           "stride": 1,
                           "padding": 1},
                 "pool2": {"kernel_size": 2,
                           "stride": 2},
                 "conv3": {"in_channel": 64,
                           "out_channels": 128,
                           "kernel_size": 3,
                           "stride": 1,
                           "padding": 1},
                 "pool3": {"kernel_size": 2,
                           "stride": 2},
                 "conv4": {"in_channel": 128,
                           "out_channels": 256,
                           "kernel_size": 3,
                           "stride": 1,
                           "padding": 1},
                 #"pool4": {"kernel_size": 2,
                 #          "stride": 2},
                 "fc1": {"in_neuron": 4*4*128,
                        "out_neuron": 4*4*128},
                 "fc2": {"in_neuron": 4*4*128,
                         "out_neuron": 128*4}}
            self.global_model = CNN4Layer(
                self.example_shape,
                self.class_no, **model_params)
            self.global_model.initial_layers()
        elif self.model_type == consts.ResNet18_MODEL:
            self.global_model = \
                ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2],
                       num_classes=self.class_no, base_width=64)
        elif self.model_type == consts.ResNet34_MODEL:
            self.global_model = \
                ResNet(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3],
                       num_classes=self.class_no, base_width=64)
        elif self.model_type == consts.ResNet50_MODEL:
            self.global_model = \
                ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3],
                       num_classes=self.class_no, base_width=64)
        elif self.model_type == consts.ResNet101_MODEL:
            self.global_model = \
                ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3],
                       num_classes=self.class_no, base_width=64)
        elif self.model_type == consts.ResNet152_MODEL:

            self.global_model = \
                ResNet(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3],
                       num_classes=self.class_no, base_width=64)
        else:
            print("Error: There is no model named %s" % self.model_type)
            exit(1)

        self.global_model.to(**self.sys_setup)
        self.global_model.eval()
        self.present_network_structure()

    def get_client_order(self):
        # randomly generate the order of clients
        each_client_no = int(self.round_no * self.client_ratio)
        clients_order = [i for i in range(self.client_no)] * each_client_no
        clients_order = np.array(clients_order)
        np.random.shuffle(clients_order)
        clients_order = clients_order.reshape(self.round_no, -1)
        return clients_order

    def train_model(self, fl_clients, is_attack=True):
        clients_order = self.get_client_order()
        client_train_no = int(self.client_no * self.client_ratio)
        test_accuracies = list()

        for round_order in range(self.round_no):
            local_models_params, training_example_no_set = list(), list()
            chosen_clients = clients_order[round_order]

            for chosen_client_index in chosen_clients:
                if not is_attack:
                    local_model_param, local_gradients, example_no, \
                        selected_grad_ratio = \
                        fl_clients[chosen_client_index].train_model(
                            self.global_model, self.epoch_no, self.lr,
                            clip_norm=self.sys_args.clip_norm,
                            center_radius=self.center_radius_stats)
                    local_models_params.append(local_model_param)
                    training_example_no_set.append(example_no)
                    del fl_clients[chosen_client_index].local_model
                    del fl_clients[chosen_client_index].local_model_bak
                else:
                    for i in range(1):
                        ground_truth, label, target_id = \
                            self.select_attack_targets(self.sys_args.num_images)
                        local_model_param, local_gradients, example_no, \
                            selected_grad_ratio = \
                            fl_clients[chosen_client_index].train_model(
                                self.global_model, self.epoch_no, self.lr,
                                clip_norm=self.sys_args.clip_norm,
                                center_radius=self.center_radius_stats,
                                target_ground_truth=ground_truth,
                                target_label=label)

                        global start_time
                        start_time = time.time()
                        if self.sys_args.epoch_no > 1:
                            recon_result, stats, opt_score,  = \
                                self.invert_gradient_attack_epochs(
                                label, local_gradients)
                        else:
                            recon_result, stats, opt_score, = \
                                self.invert_gradient_attack_epoch1(
                                    label, local_gradients)

                        print("Grad ratio: %s" % selected_grad_ratio)
                        self.save_reconstruction_example(
                            ground_truth, recon_result, opt_score, target_id,
                            selected_grad_ratio)

                        print("Info: Attack finished!")
                        with open(consts.INFO_FILE, "a+") as f:
                            output_str = f'Info: Attack finished!.\n'
                            f.write(output_str)
                            f.write("--------------------------------------\n\n")

                    exit(1)
            self.aggregate_global_model(local_models_params,
                                        training_example_no_set,
                                        client_train_no)
            with torch.no_grad():
                acc = self.compute_accuracy()
                acc.cpu()
                test_accuracies.append(acc.tolist())
                if (round_order+1) % 5 == 0:
                    print("Round %s: Accuracy %.4f" % (round_order+1, acc))
        self.save_prediction_results(test_accuracies, selected_grad_ratio)

    def launch_GLA_attack(self):
        print("Launch inverting attack:")
        ground_truth, label = self.select_attack_targets()
        gradient = self.get_gradients_from_global_model(ground_truth, label)
        recon_result = self.invert_gradient_attack_epoch1(
            ground_truth, label, gradient)
        self.save_reconstruction_example(
            ground_truth, recon_result, label)

    def aggregate_global_model(self, local_models_params,
                               training_example_no_set, client_train_no):
        weight_keys = list(local_models_params[0].keys())
        fed_state_dict = collections.OrderedDict()
        training_example_no = sum(training_example_no_set)
        for key in weight_keys:
            key_sum = 0
            for k in range(client_train_no):
                client_data_ratio = \
                    training_example_no_set[k] / training_example_no
                key_sum += client_data_ratio * local_models_params[k][key]
            fed_state_dict[key] = key_sum
        self.global_model.load_state_dict(fed_state_dict)
        self.get_center_radius_of_model()

    def save_prediction_results(self, results, grad_ratio):
        exp_details = {"client_no": self.client_no,
                       "client_ratio": self.client_ratio,
                       "perturb": self.sys_args.perturb_mechanism,
                       "privacy budget": self.sys_args.privacy_budget,
                       "broken probability": self.sys_args.broken_probability,
                       "noise dist": self.sys_args.noise_dist,
                       "sigma": self.sys_args.sigma,
                       "noise_dist": self.sys_args.noise_dist,
                       "grad_ratio": grad_ratio,
                       "epoch": self.sys_args.epoch_no,
                       "batch size": self.sys_args.batch_size,
                       "lr": self.sys_args.lr,
                       "dataset": self.sys_args.dataset,
                       "model name": self.sys_args.model_name,
                       "clip norm": self.sys_args.clip_norm,
                       "is iid": self.sys_args.is_iid,
                       "is balanced": self.sys_args.is_balanced,
                       "results": results}
        with open(consts.EXP_RESULT_DIR, "a") as f:
            json.dump(exp_details, f, indent=4)

    def get_model_shape(self):
        if self.global_model is None:
            print("Error: The local model is Null!")
            exit(1)
        else:
            self.model_shape = dict()
            for name, param in self.global_model.state_dict().items():
                self.model_shape[name] = param.shape

    def get_center_radius_of_model(self):
        self.center_radius_stats = dict()
        weights = deepcopy(self.global_model.state_dict())
        for name, params in self.model_shape.items():
            self.center_radius_stats[name] = list()
            self.center_radius_stats[name] = \
                self.get_center_radius_of_vector(weights[name])
        #print("Info: Success to Update the center and the radius of the "
        #      "weights in the model.")

    def get_center_radius_of_vector(self, value_vector):
        """
        :param value_vector: tensor array
        :return:
        """
        max_value = value_vector.max()
        min_value = value_vector.min()
        radius_v = (max_value - min_value) / 2.0
        center_v = min_value + radius_v
        return (center_v, radius_v)

    def present_network_structure(self):
        paras = list(self.global_model.parameters())
        print("------------- Model Structure -------------")
        for num, para in enumerate(paras):
            para_size = para.size()
            print("%s: %s" % (num, para_size))
        print("------------------- END -------------------")

    def compute_accuracy(self):
        pred_r = torch.argmax(self.global_model(self.test_examples), dim=-1)
        return sum(pred_r == self.test_labels)/self.test_example_no

    def select_attack_rounds(self):
        """
        select attack round and targets.
        """
        # select attack round
        if self.attack_no is None:
            self.attack_no = 1
            self.attack_rounds = [0]
        else:
            while len(self.attack_rounds) < self.attack_no:
                attack_round_id = np.random.randint(self.round_no)
                if attack_round_id not in self.attack_rounds:
                    self.attack_rounds.append(attack_round_id)
            self.attack_rounds.sort()

    def select_attack_targets(self, target_no=1):
        if target_no == 1:
            target_ids = np.random.randint(len(self.valid_loader.dataset))
            target_ids = 6132

            print("Info: Attack target ID - %s" % target_ids)
            with open(consts.INFO_FILE, "a+") as f:
                output_str = f'\n Info: Attack target ID - {target_ids}.\n'
                f.write(output_str)

            ground_truth, label = self.valid_loader.dataset[target_ids]
            ground_truth, label = \
                (ground_truth.unsqueeze(0).to(**self.sys_setup),
                 torch.as_tensor((label,), device=self.sys_setup["device"]),)
            return ground_truth, label, target_ids
        else:
            target_ids, ground_truth, labels = [], [], []
            target_ids = [1607, 7865, 5976]

            if not target_ids:
                while len(target_ids) < target_no:
                    target_id = np.random.randint(len(self.valid_loader.dataset))
                    if target_id not in target_ids:
                        img, label = self.valid_loader.dataset[target_id]
                        labels.append(torch.as_tensor(
                            (label,), device=self.sys_setup["device"]))
                        ground_truth.append(img.to(**self.sys_setup))
                        target_ids.append(target_id)
            else:
                for target_id in target_ids:
                    img, label = self.valid_loader.dataset[target_id]
                    labels.append(torch.as_tensor(
                        (label,), device=self.sys_setup["device"]))
                    ground_truth.append(img.to(**self.sys_setup))

            ground_truth = torch.stack(ground_truth)
            labels = torch.cat(labels)
            print("Info: Attack targets ID - %s" % target_ids)
            with open(consts.INFO_FILE, "a+") as f:
                output_str = \
                    f'\n Info: Attack targets ID -{target_ids}.\n'
                f.write(output_str)
            return ground_truth, labels, target_ids

    def init_target_example(self, ground_truth, label):
        if self.sys_args.demo_target:  # demo image
            # Specify PIL filter for lower pillow versions
            ground_truth = torch.as_tensor(
                np.array(Image.open(consts.ATTACK_EXAMPLE_DEMO_DIR).resize(
                    (self.example_row_pixel, self.example_column_pixel),
                    Image.BICUBIC)) / 255, **self.sys_setup
            )
            ground_truth = ground_truth.permute(2, 0, 1).sub(self.dm) \
                .div(self.ds).unsqueeze(0).contiguous()
            if not self.sys_args.label_flip:
                label = torch.as_tensor((1,), device=self.sys_setup["device"])
            else:
                label = torch.as_tensor((5,), device=self.sys_setup["device"])
        else:
            #target_example_id = np.random.randint(target_client.example_no)
            #ground_truth, label = \
            #    target_client.get_example_by_index(target_example_id)

            if self.sys_args.label_flip:
                label = (label + 1) % self.class_no
            ground_truth, label = (
                ground_truth.unsqueeze(0).to(**self.sys_setup),
                torch.as_tensor((label,), device=self.sys_setup["device"]),
            )
        return ground_truth, label

    def compute_updated_parameters(self, updated_parameters):
        patched_model = MetaMonkey(self.global_model)
        patched_model_origin = deepcopy(patched_model)
        patched_model.parameters = collections.OrderedDict(
            (name, param - param_origin)
            for ((name, param), (name_origin, param_origin))
            in zip(updated_parameters.items(),
                   patched_model_origin.parameters.items()))
        return list(patched_model.parameters.values())

    def get_gradients_from_global_model(self, ground_truth,
                                        labels):
        self.global_model.zero_grad()
        loss_fn = Classification()
        target_loss, _, _ = loss_fn(self.global_model(ground_truth),
                                    labels)
        input_gradient = torch.autograd.grad(target_loss,
                                             self.global_model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]
        full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
        print(f"Full gradient norm is {full_norm:e}.")
        return input_gradient

    def invert_gradient_attack_epoch1(self, labels, input_gradient):
        config = dict(
            signed=self.sys_args.signed,
            boxed=self.sys_args.boxed,
            cost_fn=self.sys_args.cost_fn,
            indices=self.sys_args.indices,
            weights=self.sys_args.weights,
            lr=self.sys_args.lr if self.sys_args.lr is not None else 0.1,
            optim=self.sys_args.optimizer,
            restarts=self.sys_args.restarts,
            max_iterations=24_000,
            total_variation=self.sys_args.tv,
            init=self.sys_args.init,
            filter="none",
            lr_decay=True,
            scoring_choice=self.sys_args.scoring_choice,
        )
        rec_machine = GradientReconstructor(
            self.global_model, (self.dm, self.ds), config,
            num_images=len(labels))
        output, stats, opt_score = rec_machine.reconstruct(
            input_gradient, labels, img_shape=self.example_shape,
            dryrun=self.sys_args.dryrun)
        return output, stats, opt_score

    def invert_gradient_attack_epochs(self, labels, input_params):
        local_gradient_steps = self.sys_args.epoch_no
        local_lr = self.sys_args.lr

        #input_parameters = reconstruction_algorithms.loss_steps(
        #    model, ground_truth, labels, lr=local_lr,
        #    local_steps=local_gradient_steps
        #)
        # input_parameters = [p.detach() for name, p in input_params.items()]
        config = dict(
            signed=self.sys_args.signed,
            boxed=self.sys_args.boxed,
            cost_fn=self.sys_args.cost_fn,
            indices=self.sys_args.indices,
            weights=self.sys_args.weights,
            lr=1,
            optim=self.sys_args.optimizer,
            restarts=self.sys_args.restarts,
            max_iterations=24_000,
            total_variation=self.sys_args.tv,
            init=self.sys_args.init,
            filter="none",
            lr_decay=True,
            scoring_choice=self.sys_args.scoring_choice,
        )

        rec_machine = FedAvgReconstructor(
            self.global_model, (self.dm, self.ds), local_gradient_steps,
            local_lr, config, num_images=self.sys_args.num_images,
            use_updates=True
        )
        output, stats, opt_score = rec_machine.reconstruct(
            input_params, labels, img_shape=self.example_shape,
            dryrun=self.sys_args.dryrun)
        return output, stats, opt_score

    def save_reconstruction_example(self, ground_truth, output, opt_score,
                                    target_id, selected_grad_ratio=1.0):
        test_mse = (output - ground_truth).pow(2).mean().item()
        feat_mse = \
            (self.global_model(output) - self.global_model(ground_truth))\
                .pow(2).mean().item()
        test_psnr = metrics.psnr(output,
                                 ground_truth, factor=1 / self.ds)

        print("Test Mse: %s, Feat Mse: %s, Test Psnr: %s" %
              (test_mse, feat_mse, test_psnr))
        with open(consts.INFO_FILE, "a+") as f:
            output_str = f'Test Mse: {test_mse}, Feat Mse: {feat_mse}, ' \
                         f'Test Psnr: {test_psnr}.\n'
            f.write(output_str)

        rec_filename = None
        if self.sys_args.save_image:
            os.makedirs(self.sys_args.image_path, exist_ok=True)
            model_init_seed = np.random.randint(0, 2 ** 32 - 10)
            utils.set_random_seed(model_init_seed)

            # save the raw image
            ground_truth_denormalized = torch.clamp(
                ground_truth * self.ds + self.dm, 0, 1)
            raw_filename = (
                f'{model_init_seed}_{target_id}_In_{self.sys_args.dataset}'
                f'_{self.sys_args.model_name}_{self.sys_args.perturb_mechanism}'
                f'_{self.sys_args.sigma}.png')
            torchvision.utils.save_image(
                ground_truth_denormalized,
                os.path.join(consts.RE_IMG_DIR, raw_filename))

            # save the reconstructed image
            output_denormalized = torch.clamp(output * self.ds + self.dm, 0, 1)
            rec_filename = (
                f'{model_init_seed}_{target_id}_Re_{self.sys_args.dataset}'
                f'_{self.sys_args.model_name}_{self.sys_args.perturb_mechanism}'
                f'_{self.sys_args.sigma}_PSNR_{test_psnr}_Ratio'
                f'_{self.sys_args.grad_ratio}_{self.sys_args.noise_dist}'
                f'_epsilon_{self.sys_args.privacy_budget}.png'
            )
            torchvision.utils.save_image(output_denormalized, os.path.join(
                consts.RE_IMG_DIR, rec_filename))

        utils.save_to_table(
            self.sys_args.table_path,
            name=f"exps_1epoch_{self.sys_args.name}",
            dryrun=self.sys_args.dryrun,
            model=self.sys_args.model_name,
            dataset=self.sys_args.dataset,
            trained=self.sys_args.trained_model,
            epoch_no=self.sys_args.epoch_no,
            perturb_mec=self.sys_args.perturb_mechanism,
            sigma=self.sys_args.sigma,
            epsilon= self.sys_args.privacy_budget,
            clip_norm=self.sys_args.clip_norm,
            batch_size=self.sys_args.batch_size,
            num_images=self.sys_args.num_images,
            restarts=self.sys_args.restarts,
            OPTIM=self.sys_args.optim,
            cost_fn=self.sys_args.cost_fn,
            indices=self.sys_args.indices,
            weights=self.sys_args.weights,
            scoring=self.sys_args.scoring_choice,
            init=self.sys_args.init,
            tv=self.sys_args.tv,
            # rec_loss=self.sys_args["opt"],
            opt_score="{:.4f}".format(opt_score),
            psnr="{:.4f}".format(test_psnr),
            test_mse="{:.4f}".format(test_mse),
            feat_mse="{:.4f}".format(feat_mse),
            target_id=target_id,
            # seed=model_seed,
            timing=str(
                datetime.timedelta(seconds=time.time() - start_time)),
            dtype=self.sys_setup["dtype"],
            # epochs=defs.epochs,
            val_acc=selected_grad_ratio,
            rec_img=rec_filename,
            # gt_img=rec_filename,
            gt_img=self.sys_args.noise_dist
        )

    def compute_gradient_by_autograd(self, global_model, ground_truth, labels):
        local_model = deepcopy(global_model)
        loss_fn = Classification()
        local_model.zero_grad()
        target_loss, _, _ = loss_fn(local_model(ground_truth), labels)

        input_gradient = torch.autograd.grad(
            target_loss, local_model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]
        return input_gradient

