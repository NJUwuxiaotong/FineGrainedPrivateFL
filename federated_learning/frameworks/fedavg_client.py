import collections
import copy
import math
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

import torch

from constant import consts
from copy import deepcopy
from loss import Classification

from pub_lib.pub_libs import random_value_with_probs, gradient_l2_norm


class FedAvgClient(object):
    def __init__(self, sys_setup, model_type, data_loader, data_info,
                 example_shape, class_no, loss_fn, privacy_budget, training_no,
                 perturb_mechanism, noise_dist, broken_prob, sigma, batch_size,
                 grad_ratio):
        self.sys_setup = sys_setup
        self.model_type = model_type
        self.data_loader = data_loader
        self.data_info = data_info
        self.example_shape = example_shape
        self.label_unique_no = class_no
        self.example_no = data_info.example_no

        self.channel_no = example_shape[0]
        self.training_row_pixel = example_shape[1]
        self.training_column_pixel = example_shape[2]

        self.local_model = None
        self.local_model_bak = None
        self.layer_weight_no = None
        self.model_shape = None
        self.model_shape_name = None
        self.model_param_no = 0
        self.loss_fn = loss_fn
        self.epoch_total_loss = 0.0

        self.perturb_mechanism = perturb_mechanism
        self.privacy_budget = privacy_budget
        self.broken_prob = broken_prob
        self.training_no = training_no
        self.single_privacy_cost = (privacy_budget + 0.0) / self.training_no
        self.noise_dist = noise_dist
        self.sigma = sigma
        self.batch_size = batch_size
        self.grad_ratio = grad_ratio

    def train_model(self, global_model, epoch_no=10, lr=0.001,
                    clip_norm=None, center_radius=None,
                    target_ground_truth=None,
                    target_label=None):
        self.local_model = copy.deepcopy(global_model)
        self.local_model_bak = copy.deepcopy(global_model)
        self.local_model.to(**self.sys_setup)
        self.local_model.eval()
        self.get_model_shape()

        if self.perturb_mechanism in consts.ALGs_GradSGD_OPT:
            return self.train_model_with_gradient_sgd(
                epoch_no, lr, clip_norm, None, self.privacy_budget,
                self.broken_prob, self.perturb_mechanism,
                target_ground_truth, target_label)
        elif self.perturb_mechanism in consts.ALGs_GradBatchOPT:
            return self.train_model_with_gradient_mini_batch_gd(
                epoch_no, lr, clip_norm, self.sigma, self.privacy_budget,
                self.broken_prob, self.perturb_mechanism,
                target_ground_truth, target_label)
        elif self.perturb_mechanism in consts.ALGs_Weight_OPT:
            return self.train_model_with_weight(
                epoch_no, lr, self.sigma, self.privacy_budget, self.broken_prob,
                self.perturb_mechanism, center_radius,
                target_ground_truth, target_label)
        elif self.perturb_mechanism in consts.ALGs_Sample_OPT:
            return self.train_model_with_sample(
                self.local_model, epoch_no, lr, clip_norm, self.sigma,
                self.privacy_budget, self.broken_prob, self.perturb_mechanism,
                target_ground_truth, target_label)
        else:
            print("Error: Perturbation mechanism %s does not exist!"
                  % self.perturb_mechanism)
            exit(1)

    def train_model_with_gradient_sgd(
            self, epoch_no, lr, norm_bound, sigma, epsilon, delta, perturb_mec,
            target_ground_truth=None, target_label=None):
        for epoch in range(epoch_no):
            gradients, gradients_full_norm = [], []
            if target_ground_truth is None:
                chosen_batch_index = np.random.randint(0, len(self.data_loader))
                for step, (examples, labels) in enumerate(self.data_loader):
                    if step == chosen_batch_index:
                        example_no = examples.shape[0]
                        for i in range(example_no):
                            gradient, full_norm = self.get_example_gradient(
                                examples[i], labels[i], 1)
                            gradients.append(gradient)
                            gradients_full_norm.append(full_norm)
            else:
                for i in range(len(target_label)):
                    gradient, full_norm = self.get_example_gradient(
                        target_ground_truth[i], target_label[i], 1)
                    gradients.append(gradient)
                    gradients_full_norm.append(full_norm)
            updated_gradient = self.update_model_avg_gradients_noise(
                gradients, gradients_full_norm, perturb_mec, lr, epsilon,
                delta, norm_bound, sigma)

        if epoch_no == 1:
            output_grad = updated_gradient
        else:
            output_grad = self.compute_updated_gradients_no_name(
                self.local_model_bak.state_dict(),
                self.local_model.state_dict(), -1)
        return self.local_model.state_dict(), output_grad, len(gradients), 1.0

    def train_model_with_gradient_mini_batch_gd(
            self, epoch_no, lr, norm_bound, sigma, epsilon, delta, perturb_mec,
            target_ground_truth=None, target_label=None):
        opt = torch.optim.SGD(self.local_model.parameters(), lr=lr)
        for epoch in range(epoch_no):
            gradients = []
            gradients_full_norm = []
            if target_ground_truth is None:
                example_no = self.example_no
                for step, (examples, labels) in enumerate(self.data_loader):
                    example_no = examples.shape[0]
                    # import pdb; pdb.set_trace()
                    gradient, full_norm = self.get_example_gradient(
                        examples, labels, example_no)
                    gradients.append(gradient)
                    gradients_full_norm.append(full_norm)

                    self.update_model_avg_gradients_noise([gradient], [full_norm], perturb_mec, lr, epsilon, delta, norm_bound, sigma)

            else:
                example_no, start_pos = len(target_label), 0
                if example_no > 1:
                    batch_size = 2
                    batch_no = example_no // batch_size
                else:
                    batch_size, batch_no = 1, 1
                for i in range(batch_no):
                    gradient, full_norm = self.get_example_gradient(
                        target_ground_truth[start_pos: start_pos + batch_size],
                        target_label[start_pos: start_pos + batch_size],
                        batch_size)
                    start_pos += batch_size
                    gradients.append(gradient)
                    gradients_full_norm.append(full_norm)
            #self.update_model_avg_gradients_noise(
            #    gradients, gradients_full_norm, perturb_mec, lr, epsilon,
            #    delta, norm_bound, sigma)

        output_grad = self.compute_updated_gradients_no_name(
            self.local_model_bak.state_dict(),
            self.local_model.state_dict(), -1)
        return self.local_model.state_dict(), output_grad, example_no, 1.0

    def update_model_avg_gradients_noise(self, gradients, gradients_full_norm,
                                         perturb_mec, lr, epsilon, delta,
                                         norm_bound, sigma):
        if sigma is None:
            sigma = self.get_sigma_of_gaussian(perturb_mec, epsilon,
                                               delta, norm_bound)
        # sigma = self.sigma * norm_bound
        sigma = self.sigma
        # print("Perturb Mec: %s, Sigma: %.6f" % (perturb_mec, sigma))
        with open(consts.INFO_FILE, "a+") as f:
            output_str = f'Perturb Mec: {perturb_mec}, Sigma: {sigma}.\n'
            f.write(output_str)

        noise = self.generate_noise(consts.GAUSSIAN_DIST, sigma,
                                    self.max_dimen)
        noise = torch.tensor(noise, device=self.sys_setup["device"])
        updated_gradient = [torch.zeros_like(layer_gradient)
                            for layer_gradient in gradients[0]]

        for i in range(len(gradients)):
            for j in range(len(updated_gradient)):
                if perturb_mec == consts.ALG_NoGradSGD:
                    updated_gradient[j] += gradients[i][j]
                elif perturb_mec == consts.ALG_rGaussAGrad18:
                    layer_norm_bound = norm_bound
                    updated_gradient[j] += gradients[i][j] * \
                            min(1, layer_norm_bound / gradients_full_norm[i])  # layer_norm_bound
                    vector_shape = updated_gradient[j].shape
                    updated_gradient[j] += noise[:vector_shape[-1]]
                else:
                    # import pdb; pdb.set_trace()

                    updated_gradient[j] += gradients[i][j] * min(
                        1, norm_bound / gradients_full_norm[i])
                    vector_shape = updated_gradient[j].shape
                    updated_gradient[j] += noise[:vector_shape[-1]]


        for i in range(len(updated_gradient)):
            updated_gradient[i] /= len(gradients)

        local_model_params = self.update_gradient_of_local_model(
            updated_gradient, lr)
        self.local_model.load_state_dict(local_model_params)
        return updated_gradient

    def get_sigma_of_gaussian(self, perturb_mec, epsilon, delta,
                              norm_bound=1.0):
        if perturb_mec == consts.ALG_rGaussAGrad16:
            sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
            sigma *= norm_bound
        elif perturb_mec == consts.ALG_eGaussAGrad19:
            s = math.log(math.sqrt(2.0 / math.pi) / delta)
            sigma = (math.sqrt(s) + math.sqrt(s + epsilon)) / (
                        math.sqrt(2) * epsilon)
        elif perturb_mec == consts.ALG_eGaussAGrad22:
            sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        elif perturb_mec == consts.ALG_rGaussPGrad22:
            sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon / math.sqrt(
                0.1)
        elif perturb_mec == consts.ALG_rGaussAGrad18:
            sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        elif perturb_mec == consts.ALG_bGaussAWeig21:
            sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        elif perturb_mec == consts.ALG_rGaussAWeig19:
            sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        elif perturb_mec == consts.ALG_rExpPWeig20:
            sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        elif perturb_mec == consts.ALG_rGaussPGrad22:
            sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
            sigma = sigma * norm_bound / math.sqrt(100)
        else:
            # consts.ALG_NoGradSGD
            sigma = 1.0
        return sigma

    def update_model_gradient_weight(self, opt, examples, labels, sigma,
                                     epsilon, delta, perturb_mec):
        if len(labels) == 1:
            examples_shape = examples.shape
            if len(examples_shape) < 4:
                examples, labels = \
                    (examples.unsqueeze(0).to(**self.sys_setup),
                     torch.as_tensor((labels,), device=self.sys_setup["device"]),)
        else:
            examples = examples.to(self.sys_setup["device"])
            labels = labels.to(self.sys_setup["device"])

        opt.zero_grad()
        loss, _, _ = self.loss_fn(self.local_model(examples), labels)
        loss.backward()
        opt.step()

        if perturb_mec == consts.ALG_bGaussAWeig21:
            if sigma is None:
                sigma = self.get_sigma_of_gaussian(
                    perturb_mec, epsilon, delta)
            sigma = self.sigma
            updated_params = self.add_dynamic_noise_to_model(
                self.local_model, consts.GAUSSIAN_DIST, sigma)
            self.local_model.load_state_dict(updated_params)

    def train_model_with_weight(self, epoch_no, lr, sigma, epsilon, delta,
                                perturb_mec, center_radius_stats=None,
                                target_ground_truth=None, target_label=None):
        opt = torch.optim.SGD(self.local_model.parameters(), lr=lr)
        #print("Perturb Mec: %s, Sigma: %s" % (self.perturb_mechanism,
        #                                      self.sigma))
        for epoch in range(epoch_no):
            if target_ground_truth is not None:
                example_no, start_pos = len(target_label), 0
                if example_no > 1:
                    batch_size = 2
                    batch_no = example_no // batch_size
                else:
                    batch_no, batch_size = 1, 1

                for i in range(batch_no):
                    self.update_model_gradient_weight(
                        opt,
                        target_ground_truth[start_pos: start_pos + batch_size],
                        target_label[start_pos: start_pos + batch_size],
                        sigma, epsilon,
                        delta, perturb_mec)
                    start_pos += batch_size
            else:
                for step, (examples, labels) in enumerate(self.data_loader):
                    examples = examples.to(self.sys_setup["device"])
                    labels = labels.to(self.sys_setup["device"])

                    self.update_model_gradient_weight(
                        opt, examples, labels, sigma, epsilon, delta,
                        perturb_mec)

        if perturb_mec == consts.ALG_rGaussAWeig19:
            if self.noise_dist == consts.LAPLACE_DIST:
                sigma = 1 / epsilon
                updated_params = self.add_dynamic_noise_to_model(
                    self.local_model, consts.LAPLACE_DIST, sigma)
            elif self.noise_dist == consts.GAUSSIAN_DIST:
                sigma = self.get_sigma_of_gaussian(perturb_mec, epsilon, delta)
                sigma = self.sigma
                updated_params = self.add_dynamic_noise_to_model(
                    self.local_model, consts.GAUSSIAN_DIST, sigma)
            else:
                print("Error: No distribution %s" % self.noise_dist)
                exit(1)
            self.local_model.load_state_dict(updated_params)

        if perturb_mec == consts.ALG_rRResAWeig21:
            if center_radius_stats is None:
                print("Error: Perturb Mechanism %s needs central and radius "
                      "info!" % perturb_mec)
                exit(1)

            updated_params = self.add_bernoulli_noise_to_model(
                self.local_model, center_radius_stats, epsilon)
            self.local_model.load_state_dict(updated_params)

        output_grad = self.compute_updated_gradients_no_name(
            self.local_model_bak.state_dict(),
            self.local_model.state_dict(), 1)
        return self.local_model.state_dict(), output_grad, \
            self.data_info.example_no, 1.0

    def train_model_with_sample(self, global_model, epoch_no, lr, norm_bound,
                                sigma, epsilon, delta, perturb_mec,
                                target_ground_truth=None, target_label=None):
        global_model_params = copy.deepcopy(global_model.state_dict())
        #init_model_params = copy.deepcopy(global_model.state_dict())

        if target_ground_truth is not None:
            updated_model_params, gradients, example_no, _ = \
                self.train_model_with_gradient_sgd(
                    epoch_no, lr, None, sigma, epsilon, delta,
                    consts.ALG_NoGradSGD, target_ground_truth, target_label)
        else:
            updated_model_params, gradients, example_no, _ = \
                self.train_model_with_gradient_sgd(epoch_no, lr, None, sigma,
                                                   epsilon, delta,
                                                   consts.ALG_NoGradSGD)

        #updated_gradients = self.compute_updated_parameters(
        #    init_model_params, updated_model_params, -1.0 * lr)
        if perturb_mec in [consts.ALG_rLapPGrad15, consts.ALG_NO_rLapPGrad15]:
            noise_gradients, selected_grad_no = \
                self.train_model_with_rLapPGrad15(
                    gradients, 1.0, self.grad_ratio, self.perturb_mechanism)
        elif perturb_mec in [consts.ALG_rExpPWeig20, consts.ALG_No_rExpPWeig20]:
            noise_gradients = self.train_model_with_rExpPWeig20(
                gradients, self.privacy_budget, 0.5,
                self.perturb_mechanism)
        elif perturb_mec in \
                [consts.ALG_rGaussPGrad22, consts.ALG_No_rGaussPGrad22]:
            noise_gradients, selected_grad_no = \
                self.train_model_with_rGaussPGrad22(
                    gradients, self.grad_ratio, norm_bound, sigma, epsilon,
                    delta, self.perturb_mechanism)
        else:
            noise_gradients = copy.deepcopy(gradients)
            selected_grad_no = 1

        grad_index = 0
        for name, params in global_model_params.items():
            if "weight" in name or "bias" in name:
                global_model_params[name] += noise_gradients[grad_index]
                grad_index += 1
        # print("Ratio: %s" % selected_grad_no)
        return global_model_params, noise_gradients, example_no, \
            selected_grad_no

    def train_model_with_rLapPGrad15(self, gradients, g_threshold, g_range,
                                     perturb_mec):
        # pre_privacy_cost = self.privacy_budget * 8.0 / 9
        perturb_privacy_cost = self.privacy_budget * 1.0 / 9
        noise_gradient = [copy.deepcopy(grad.reshape((1, -1))[0])
                          for grad in gradients]
        grad_no, selected_grad_no = 0.0, 0.0
        # tmp_gradient = copy.deepcopy(noise_gradient)
        for i in range(len(gradients)):
            layer_grad_no = len(noise_gradient[i])
            #pre_noises = torch.tensor(
            #    self.generate_noise(consts.LAPLACE_DIST,
            #                        2 / pre_privacy_cost, layer_grad_no),
            #    device=self.sys_setup["device"])
            #tmp_noises = torch.tensor(
            #    self.generate_noise(consts.LAPLACE_DIST,
            #                        4 / perturb_privacy_cost, layer_grad_no),
            #    device=self.sys_setup["device"])
            suffix_noises = torch.tensor(
                self.generate_noise(consts.LAPLACE_DIST,
                                    2 / perturb_privacy_cost, layer_grad_no),
                device=self.sys_setup["device"])
            grad_no += layer_grad_no

            #tmp_gradient[i][tmp_gradient[i] > g_threshold] = g_threshold
            #tmp_gradient[i][tmp_gradient[i] < -1.0 * g_threshold] = \
            #    -1.0 * g_threshold
            #import pdb; pdb.set_trace()
            # selected_pos = \
            #     tmp_gradient[i] + tmp_noises < pre_noises + g_range
            #noise_gradient[i][noise_gradient[i] > g_threshold] = g_threshold
            #noise_gradient[i][noise_gradient[i] < -1.0 * g_threshold] = \
            #    -1.0 * g_threshold
            if perturb_mec == consts.ALG_rLapPGrad15:
                noise_gradient[i] += suffix_noises
                raw_layer_grads = \
                    copy.deepcopy(noise_gradient[i]).reshape(1, -1)[0]
                value_top_k = self.get_k_th_value(raw_layer_grads, g_range)
                selected_pos_index = noise_gradient[i] < value_top_k
                noise_gradient[i][selected_pos_index] = 0.0
                selected_grad_no += selected_pos_index.sum()

                # selected_pos = torch.tensor([False] * layer_grad_no,
                #                             device=self.sys_setup["device"])
                # selected_pos_index = torch.tensor(
                #    random.sample(range(layer_grad_no),
                #                   int(layer_grad_no * (1-g_range))),
                #     device=self.sys_setup["device"])
                # if len(selected_pos_index) > 0:
                # selected_pos[selected_pos_index] = True

        for i in range(len(noise_gradient)):
            noise_gradient[i] = noise_gradient[i].reshape(gradients[i].shape)

        #print("Grad Ratio: %s / %s = %s" %
        #      (grad_no - selected_grad_no, grad_no,
        #       (grad_no - selected_grad_no) / grad_no))
        return noise_gradient, ((grad_no - selected_grad_no) / grad_no).tolist()

    def train_model_with_rExpPWeig20(self, gradients, epsilon,
                                     total_dimen_ratio, perturb_mec):
        privacy_cost1 = 1.0 / 2 * epsilon
        privacy_cost2 = epsilon - privacy_cost1

        dimen_list = list(self.layer_weight_no.values())
        chosen_dimens_index = self.select_dimens_exp(epsilon,
                                                     total_dimen_ratio)
        sigma = self.get_sigma_of_gaussian(perturb_mec,
                                           privacy_cost1, self.broken_prob)
        sigma = 0.0001
        noise_gradient = collections.OrderedDict(
            [(name, torch.zeros_like(layer_gradient))
             for name, layer_gradient in gradients.items()])
        # print("Perturb mec: %s, Sigma: %s" % (self.perturb_mechanism, self.sigma))
        for dimen_index in chosen_dimens_index:
            noise = self.generate_noise(consts.LAPLACE_DIST, self.sigma,
                                        dimen_list[dimen_index])
            name = self.model_shape_name[dimen_index]
            noise_tensor = \
                torch.tensor(noise, device=self.sys_setup["device"]).reshape(
                    gradients[name].shape)
            if perturb_mec == consts.ALG_rExpPWeig20:
                noise_gradient[name] = noise_gradient[name] + gradients[name] \
                                       + noise_tensor
            else:
                noise_gradient[name] = gradients[name]
        return noise_gradient

    def train_model_with_rGaussPGrad22(self, gradients, top_k_ratio, norm_bound,
                                       sigma, epsilon, delta, perturb_mec):
        grad_no, selected_grad_no, noisy_gradient = 0.0, 0.0, list()
        l2_norm, layer_l2_norm = gradient_l2_norm(gradients)
        # sigma = self.get_sigma_of_gaussian(perturb_mec, epsilon, delta,
        #                                    norm_bound)
        for layer_grads in gradients:
            raw_layer_grads = copy.deepcopy(layer_grads).reshape(1, -1)[0]
            noise_values = copy.deepcopy(raw_layer_grads)
            noise = torch.tensor(
                self.generate_noise(consts.GAUSSIAN_DIST, sigma,
                                    len(raw_layer_grads)),
                device=self.sys_setup["device"])
            value_top_k = self.get_k_th_value(raw_layer_grads, top_k_ratio)
            noise_values[raw_layer_grads < value_top_k] = 0.0
            noise[raw_layer_grads < value_top_k] = 0.0

            grad_no += len(raw_layer_grads)
            selected_grad_no += (raw_layer_grads < value_top_k).sum()
            #print("s: %s/%s = %s" % (
            #    (raw_layer_grads < value_top_k).sum(), len(raw_layer_grads),
            #    (raw_layer_grads < value_top_k).sum()/len(raw_layer_grads)))
            if perturb_mec == consts.ALG_rGaussPGrad22:
                noisy_gradient.append(
                    (noise_values * min(1, norm_bound / l2_norm) +  noise)\
                        .reshape(layer_grads.shape))
            else:
                noisy_gradient.append(layer_grads)

        #print("s: %s/%s = %s" % (
        #    (grad_no - selected_grad_no, grad_no,
        #    (grad_no - selected_grad_no)/ grad_no)))
        return noisy_gradient, ((grad_no - selected_grad_no)/ grad_no).tolist()

    def get_example_by_index(self, example_id):
        return self.data_loader.dataset[example_id]

    def get_k_th_value(self, tensor_values, top_k_ratio):
        raw_values = copy.deepcopy(tensor_values).tolist()
        raw_values.sort()
        raw_values.reverse()
        if top_k_ratio < 1:
            value_top_k = raw_values[int(len(raw_values) * top_k_ratio)]
        else:
            value_top_k = raw_values[-1] - 1
        return value_top_k

    def get_example_gradient(self, ground_truth, label, exp_no):
        if exp_no == 1:
            if len(ground_truth.shape) < 4:
                ground_truth, label = \
                    (ground_truth.unsqueeze(0).to(**self.sys_setup),
                     torch.as_tensor((label,), device=self.sys_setup["device"]),)
        else:
            ground_truth = ground_truth.to(self.sys_setup["device"])
            label = label.to(self.sys_setup["device"])

        self.local_model.zero_grad()
        target_loss, _, _ = self.loss_fn(
            self.local_model(ground_truth), label)

        # import pdb; pdb.set_trace()
        try:
            input_gradient = torch.autograd.grad(
                target_loss, self.local_model.parameters())
        except:
            import pdb; pdb.set_trace()

        input_gradient = [grad.detach() for grad in input_gradient]
        full_norm = torch.stack(
            [g.norm() for g in input_gradient]).mean()
        # print(f"Full gradient norm is {full_norm:e}.")
        del ground_truth
        del label
        torch.cuda.empty_cache()
        return input_gradient, full_norm

    def select_dimens_exp(self, privacy_budget, total_dimen_ratio):
        dimen_list = list(self.layer_weight_no.values())
        dimen_status_vector = np.array(dimen_list).argsort().tolist()

        dimen_probs = list()
        for i in range(len(dimen_list)):
            dimen_index = dimen_status_vector.index(i)
            prob = math.exp(privacy_budget *
                            (dimen_index + 1) / (dimen_list[i] - 1))
            dimen_probs.append(prob)

        dimen_probs = (np.array(dimen_probs) / sum(dimen_probs)).tolist()
        chosen_dimens_index = random_value_with_probs(
            dimen_probs, int(len(dimen_list) * total_dimen_ratio))
        return chosen_dimens_index

    def add_dynamic_noise_to_model(self, local_model, noise_dist, sigma):
        origin_model = copy.deepcopy(local_model.state_dict())
        updated_parames = list()
        with torch.no_grad():
            for name, param in origin_model.items():
                noises = self.generate_noise(noise_dist, sigma,
                                             self.layer_weight_no[name])
                noises = torch.tensor(noises, device=self.sys_setup['device'])
                noises = noises.reshape(self.model_shape[name])
                updated_parames.append((name, param + noises))
        return collections.OrderedDict(updated_parames)

    def add_bernoulli_noise_to_model(self, local_model,
                                     center_radius_of_weights, epsilon):
        model_params = copy.deepcopy(local_model.state_dict())
        updated_parames = list()
        for name, params in model_params.items():
            layer_params = params.reshape((1, -1))[0]
            noise_layer_params = torch.zeros_like(layer_params)
            center_v = center_radius_of_weights[name][0]
            radius_v = center_radius_of_weights[name][1]
            if radius_v == 0.0:
                radius_v = 0.000001
            try:
                rr_value = (math.exp(self.privacy_budget) + 1) / \
                       (math.exp(self.privacy_budget) - 1)
            except:
                rr_value = 1.0
            w1 =  center_v + rr_value * radius_v
            w2 = center_v - rr_value * radius_v
            noise_layer_params = [w1] * len(noise_layer_params)
            noise_layer_params = torch.tensor(noise_layer_params,
                                              device=self.sys_setup["device"])
            probs = 1 / rr_value * (layer_params - center_v) / 2 / radius_v + 0.5
            probs[probs > 1.0] = 1.0
            probs[probs < 0.0] = 0.0
            try:
                bernoulli_values = bernoulli.rvs(probs.tolist())
            except:
                import pdb; pdb.set_trace()
            noise_layer_params[bernoulli_values == 0] = w2
            noise_layer_params = noise_layer_params.reshape(params.shape)
            updated_parames.append((name, noise_layer_params))
        return collections.OrderedDict(updated_parames)

    def bernoulli_noise(self, weight, privacy_budget, center_v,
                                 radius_v):
        prob = ((weight - center_v)*(math.exp(privacy_budget) - 1) +
                radius_v *(math.exp(privacy_budget) + 1)) / \
               (2*radius_v*(math.exp(privacy_budget) + 1))
        try:
            random_v = bernoulli.rvs(prob.tolist())
        except:
            random_v = 0
        if random_v == 1:
            return center_v + \
                   radius_v * (math.exp(privacy_budget) + 1) / \
                   (math.exp(privacy_budget) - 1)
        else:
            return center_v - \
                   radius_v * (math.exp(privacy_budget) + 1) / \
                   (math.exp(privacy_budget) - 1)

    def compute_updated_parameters(self, initial_parameters,
                                   updated_parameters, lr):
        fed_state_dict = collections.OrderedDict()
        for ((name, param), (name_origin, param_origin)) in zip(
                updated_parameters.items(), initial_parameters.items()):
            if "weight" in name or "bias" in name:
                fed_state_dict[name] = -1.0* (param_origin - param) / lr
        return fed_state_dict

    def compute_updated_gradients_no_name(self, initial_parameters,
                                          updated_parameters, lr=1):
        updated_gradients = list()
        for ((name, param), (name_origin, param_origin)) in zip(
                updated_parameters.items(), initial_parameters.items()):
            if "weight" in name or "bias" in name:
                updated_gradients.append((param_origin - param)/lr)
        return updated_gradients

    def generate_noise(self, noise_dist, lap_sigma, noise_no):
        if noise_dist == consts.LAPLACE_DIST:
            return np.random.laplace(0, lap_sigma, noise_no)
        elif noise_dist == consts.GAUSSIAN_DIST:
            return np.random.normal(0, lap_sigma, noise_no)
        else:
            print("No distribution %s" % noise_dist)
            exit(1)

    def compute_gradient_by_autograd(self, global_model, ground_truth, labels):
        local_model = deepcopy(global_model)
        loss_fn = Classification()
        local_model.zero_grad()
        target_loss, _, _ = loss_fn(local_model(ground_truth), labels)

        input_gradient = torch.autograd.grad(
            target_loss, local_model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]
        return input_gradient

    def compute_gradient_by_opt(self, global_model, ground_truth, labels):
        local_model = deepcopy(global_model)
        local_lr = 1e-4
        updated_parameters = self.stochastic_gradient_descent(
            local_model, ground_truth, labels, 1, lr=local_lr)
        updated_gradients = self.compute_updated_parameters(
            global_model, updated_parameters, local_lr)
        updated_gradients = [p.detach() for p in updated_gradients]
        return updated_gradients

    def stochastic_gradient_descent(self, local_model, training_examples,
                                    training_labels,
                                    epoch_no=10, lr=0.001):
        opt = torch.optim.SGD(local_model.parameters(), lr)
        for epoch in range(epoch_no):
            pred_label = local_model(training_examples)
            loss = self.loss_fn(pred_label, training_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return local_model.state_dict()

    def show_hist_of_gradients(self, gradients, data_type="list"):
        if data_type == "list":
            plt.hist(gradients.tolist(), bins=1000)
        else:
            results = list()
            for name, grad in gradients.items():
                results.extend(gradients[name].reshape((1, -1))[0].tolist())
            plt.hist(results, bins=1000)
        plt.show()

    def get_model_shape(self):
        if self.local_model is None:
            print("Error: The local model is Null!")
            exit(1)
        else:
            self.model_shape, self.model_shape_name = dict(), list()
            self.layer_weight_no, self.max_dimen = dict(), 0
            for name, param in self.local_model.state_dict().items():
                self.model_shape[name] = param.shape
                self.model_shape_name.append(name)
        
                param_no = 1
                for param_n in param.shape:
                    param_no *= param_n
                self.layer_weight_no[name] = param_no
                self.model_param_no += param_no
                
                if self.max_dimen < param_no:
                    self.max_dimen = param_no

    def update_gradient_of_local_model(self, gradients, lr):
        model_params = copy.deepcopy(self.local_model.state_dict())
        grad_no = 0
        for name, params in model_params.items():
            if "weight" in name or "bias" in name:
                model_params[name] -= lr * gradients[grad_no]
                grad_no += 1
        return model_params
