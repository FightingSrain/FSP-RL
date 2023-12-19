import copy
import math
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import autograd
from torch.distributions import Categorical, Normal, Independent
from skimage.metrics import structural_similarity as ssim_cal
from skimage.metrics import peak_signal_noise_ratio as psnr_cal


from config import config


class PixelWiseA3C_InnerState():

    def __init__(self, model,
                 optimizer,
                 batch_size,
                 t_max,
                 gamma,
                 beta=1e-2,
                 phi=lambda x: x,
                 pi_loss_coef=1.0,
                 v_loss_coef=0.5,
                 average_reward_tau=1e-2,
                 act_deterministically=False,
                 average_entropy_decay=0.999,
                 average_value_decay=0.999):

        self.shared_model = model
        self.model = copy.deepcopy(self.shared_model)

        self.optimizer = optimizer
        self.batch_size = batch_size

        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.average_reward_tau = average_reward_tau
        self.act_deterministically = act_deterministically
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay


        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_continue_action_log_prob = {}
        self.past_continue_action_entropy = {}

        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}
        self.average_reward = 0

        self.explorer = None

        # Stats
        self.average_value = 0
        self.average_entropy = 0

    """
    异步更新参数
    """

    def sync_parameters(self):
        for m1, m2 in zip(self.model.modules(), self.shared_model.modules()):
            m1._buffers = m2._buffers.copy()
        for target_param, param in zip(self.model.parameters(), self.shared_model.parameters()):
            target_param.detach().copy_(param.detach())

    """
    异步更新梯度
    """

    def update_grad(self, target, source):
        target_params = dict(target.named_parameters())
        for param_name, param in source.named_parameters():
            if target_params[param_name].grad is None:
                if param.grad is None:
                    pass
                else:
                    target_params[param_name].grad = param.grad
            else:
                if param.grad is None:
                    target_params[param_name].grad = None
                else:
                    target_params[param_name].grad[...] = param.grad

    # def update_grad(self, target, source):
    #     target_params = dict(target.named_parameters())
    #     for param_name, param in source.named_parameters():
    #         if param.grad is not None:  # check if param has gradient
    #             if target_params[param_name].grad is None:  # initialize target gradient if None
    #                 target_params[param_name].grad = torch.zeros_like(param.grad)
    #             target_params[param_name].grad[...] = param.grad.clone()  # copy param gradient to target gradient
    #     target.zero_grad()  # clear previous gradients of target

    def update(self, statevar):
        assert self.t_start < self.t
        if statevar is None:
            R = torch.zeros(self.batch_size*3, 1, config.img_size, config.img_size).cuda()
        else:
            _, vout, _, _, _, _ = self.model(statevar)
            R = vout.detach()

        pi_loss = 0
        v_loss = 0

        for i in reversed(range(self.t_start, self.t)):
            R *= self.gamma
            R = self.model.conv_smooth(R)
            R += self.past_rewards[i]
            v = self.past_values[i]

            advantage = R - v.detach()
            log_prob = self.past_action_log_prob[i]
            entropy = self.past_action_entropy[i]

            continue_log_prob = self.past_continue_action_log_prob[i]
            continue_entropy = self.past_continue_action_entropy[i]

            pi_loss -= log_prob * advantage
            pi_loss -= self.beta * entropy

            pi_loss -= continue_log_prob * advantage
            pi_loss -= self.beta * continue_entropy

            v_loss += (v - R) ** 2 / 2.

        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef

        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef

        print("\npi_loss:", pi_loss.mean())
        print("\nv_loss:", v_loss.mean())
        print("==========")
        total_loss = (pi_loss + v_loss).mean()

        print("\ntotal_loss:", total_loss)

        self.optimizer.zero_grad()
        total_loss.backward()

        self.update_grad(self.shared_model, self.model)
        self.optimizer.step()
        self.sync_parameters()

        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_continue_action_log_prob = {}
        self.past_continue_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}
        self.past_valr = {}
        self.t_start = self.t = 0

    def _normal_logproba(self, x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)
        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba

    def get_action(self, state):
        statevar = torch.Tensor(state).cuda()
        pout, mean, logstd, vout = self.model(statevar)
        n, num_actions, h, w = pout.shape

        p_trans = pout.permute([0, 2, 3, 1]).contiguous().view(-1, pout.shape[1])
        dist = Categorical(p_trans)
        action = dist.sample()

        std = torch.exp(logstd)
        continue_dist = Normal(mean, std)
        continue_dist = Independent(continue_dist, 1)
        continue_act = continue_dist.sample()
        return action.view(n, h, w).detach().cpu().numpy(), \
               continue_act.detach().cpu().numpy()

    def act_and_train(self, state, ht1, ht2, reward, test=False):
        statevar = torch.Tensor(state).cuda()
        ht1 = torch.Tensor(ht1).cuda()
        ht2 = torch.Tensor(ht2).cuda()
        self.past_rewards[self.t - 1] = torch.Tensor(reward).cuda()

        if self.t - self.t_start == self.t_max:
            self.update(statevar)

        self.past_states[self.t] = statevar
        if test:
            pout, vout, mean, logstd, h_t1, h_t2 = self.model(torch.cat([statevar, ht1, ht2], 1))
            n, num_actions, h, w = pout.shape
        else:
            pout, vout, mean, logstd, h_t1, h_t2 = self.model(torch.cat([statevar, ht1, ht2], 1))
            n, num_actions, h, w = pout.shape

            p_trans = pout.permute([0, 2, 3, 1]).contiguous().view(-1, pout.shape[1])
            dist = Categorical(p_trans)
            action = dist.sample()
            log_p = torch.log(torch.clamp(p_trans, min=1e-9, max=1-1e-9))
            log_action_prob = torch.gather(log_p, 1, Variable(action.unsqueeze(-1))).view(n, 1, h, w)
            entropy = -torch.sum(p_trans * log_p, dim=-1).view(n, 1, h, w)

            std = torch.exp(logstd)

            continue_dist = Normal(mean, std.squeeze())
            continue_dist = Independent(continue_dist, 1)
            continue_act = continue_dist.sample()

            continue_acts = (continue_act.view(n, num_actions, 1, 1) *
                            torch.ones((n, num_actions,
                            h, w)).cuda()).gather(1, action.view(n, 1, h, w))
            means = (mean.view(n, num_actions, 1, 1) *
                    torch.ones((n, num_actions,
                    h, w)).cuda()).gather(1, action.view(n, 1, h, w))
            logstds = (logstd.view(n, num_actions, 1, 1) *
                     torch.ones((n, num_actions,
                     h, w)).cuda()).gather(1, action.view(n, 1, h, w))

            # means = (mean + torch.zeros_like(pout)).gather(1, action.view(n, 1, h, w))
            # logstds = (logstd + torch.zeros_like(pout)).gather(1, action.view(n, 1, h, w))
            stds = torch.exp(logstds)
            continue_log_action_prob = self._normal_logproba(continue_acts, means,
                                                             logstds, stds)
            continue_entropy = -torch.exp(continue_log_action_prob) * continue_log_action_prob

        if test:
            _, action = torch.max(pout, dim=1)
        _, tst_act = torch.max(pout, dim=1)
        tst_act = tst_act.view(n, h, w).detach().cpu()
        tst_act_par = mean.detach().cpu()


        if not test:
            self.past_action_log_prob[self.t] = log_action_prob.cuda()
            self.past_action_entropy[self.t] = entropy.cuda()
            self.past_continue_action_log_prob[self.t] = continue_log_action_prob.cuda()
            self.past_continue_action_entropy[self.t] = continue_entropy.cuda()
            self.past_values[self.t] = vout

            self.t += 1
            return action.view(n, h, w).detach().cpu().numpy(), \
                   continue_act.detach().cpu().numpy(), \
                   torch.exp(log_action_prob).squeeze(1).detach().cpu(), \
                   h_t1.detach().cpu().numpy(), \
                   h_t2.detach().cpu().numpy(), \
                   tst_act
        else:
            return tst_act.numpy(), \
                   tst_act_par.numpy(), \
                   pout.detach().cpu().numpy(), \
                   h_t1.detach().cpu().numpy(), \
                   h_t2.detach().cpu().numpy()


    def stop_episode_and_train(self, state, reward, done=False):
        self.past_rewards[self.t - 1] = torch.Tensor(reward).cuda()
        if done:
            self.update(None)
        else:
            statevar = copy.deepcopy(state)
            self.update(statevar)
    def valSIDD(self, agent, state, raw_val, raw_val_gt, test_num):
        self.model.eval()
        B, C, H, W = raw_val.shape
        val_pnsr = 0.
        val_ssim = 0.
        current_state = state.State((B, 3,
                                     config.img_tst_size, config.img_tst_size), config.MOVE_RANGE)
        reward = np.zeros((B*3, 1, config.img_size, config.img_size))

        # raw_n = np.random.normal(0, config.sigma, raw_val.shape).astype(raw_val.dtype) / 255.
        ins_noisy = np.clip(raw_val, a_min=0., a_max=1.)
        h_t1 = np.zeros([B, self.model.nf, config.img_size, config.img_size], dtype=np.float32)
        h_t2 = np.zeros([B, self.model.nf, config.img_size, config.img_size], dtype=np.float32)
        current_state.reset(ins_noisy)
        val_res = None
        for i in range(test_num):

            action, action_par, _, h_t1, h_t2 = agent.act_and_train(current_state.tensor, h_t1, h_t2,
                                                 reward,
                                                 test=True)
            current_state.step(action, action_par)
            if i == test_num - 1:
                val_res = copy.deepcopy(current_state.image)
                break

        for i in range(0, B):
            psnr = psnr_cal(raw_val_gt[i, :, :, :].transpose(1, 2, 0),
                            val_res[i, :, :, :].transpose(1, 2, 0))
            ssim = ssim_cal(raw_val_gt[i, :, :, :].transpose(1, 2, 0),
                            val_res[i, :, :, :].transpose(1, 2, 0), multichannel=True)
            val_pnsr += psnr
            val_ssim += ssim
        val_pnsr /= B
        val_ssim /= B
        self.model.train()
        return val_pnsr, val_ssim

    def val(self, agent, state, raw_val, test_num):
        self.model.eval()
        B, C, H, W = raw_val.shape
        val_pnsr = 0.
        val_ssim = 0.
        current_state = state.State((B, 3,
                                     config.img_tst_size, config.img_tst_size), config.MOVE_RANGE)
        reward = np.zeros((B*3, 1, config.img_size, config.img_size))

        # raw_n = np.random.normal(0, config.sigma, raw_val.shape).astype(raw_val.dtype) / 255.
        # ins_noisy = np.clip(raw_val + raw_n, a_min=0., a_max=1.)

        noise_level = config.sigma
        ins_noisy = torch.poisson(noise_level * torch.tensor(raw_val)) / noise_level
        ins_noisy = ins_noisy.numpy()

        h_t1 = np.zeros([B, self.model.nf, config.img_size, config.img_size], dtype=np.float32)
        h_t2 = np.zeros([B, self.model.nf, config.img_size, config.img_size], dtype=np.float32)
        current_state.reset(ins_noisy)
        val_res = None
        for i in range(test_num):

            action, action_par, _, h_t1, h_t2 = agent.act_and_train(current_state.tensor, h_t1, h_t2,
                                                 reward,
                                                 test=True)
            current_state.step(action, action_par)
            if i == test_num - 1:
                val_res = copy.deepcopy(current_state.image)
                break

        for i in range(0, B):
            psnr = psnr_cal(raw_val[i, :, :, :].transpose(1, 2, 0),
                            val_res[i, :, :, :].transpose(1, 2, 0))
            ssim = ssim_cal(raw_val[i, :, :, :].transpose(1, 2, 0),
                            val_res[i, :, :, :].transpose(1, 2, 0), multichannel=True)
            val_pnsr += psnr
            val_ssim += ssim
        val_pnsr /= B
        val_ssim /= B
        self.model.train()
        return val_pnsr, val_ssim
