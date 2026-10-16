import torch
import torch.nn.functional as F
import random
from torch.distributions import Dirichlet
import matplotlib.pyplot as plt
import numpy as np
import pygame



class ActInfAgent_02:
    def __init__(self, amount_of_states, amount_of_observations, amount_of_actions, short_term_memory, planning_horizon, forgetfulness=0.001):
        self._chosen_policy = None
        self._amount_of_states = amount_of_states
        self._amount_of_observations = amount_of_observations
        self._amount_of_actions = amount_of_actions
        self._short_term_memory = short_term_memory
        self._planning_horizon = planning_horizon
        self.action = random.randint(0, amount_of_actions-1)
        self._amount_of_policies = self._amount_of_actions ** self._planning_horizon
        self._all_action_sequences = torch.cartesian_prod(
            *[torch.arange(self._amount_of_actions)] * self._planning_horizon)
        self._observation_history = torch.empty(0, dtype=torch.long)
        self._action_history = torch.empty(0, dtype=torch.long)
        self._forgetfulness = forgetfulness

        ### Model ###

        self._past_state = torch.softmax(torch.zeros(1, self._amount_of_states), dim=-1)
        self._joint_state_state = None
        self._joint_state_observation = None
        self._future_state = torch.softmax(torch.zeros(self._amount_of_policies, self._planning_horizon,
                                                           self._amount_of_states), dim=-1)
        self._future_observation = torch.softmax(torch.zeros(self._amount_of_policies, self._planning_horizon,
                                                           self._amount_of_observations), dim=-1)
        self._prior_future_observation = torch.softmax(torch.zeros(self._amount_of_observations), dim=-1)
        self._prior_policy = torch.softmax(torch.zeros(self._amount_of_policies), dim=-1)
        self._dirichlet_alpha = 1.0 + 1.0 * torch.rand(self._amount_of_states, self._amount_of_observations)
        self._dirichlet_beta = 1.0 + 1.0 * torch.rand(self._amount_of_actions, self._amount_of_states,
                                                      self._amount_of_states)
        self._prior_likelihood = Dirichlet(self._dirichlet_alpha.clone())
        self._prior_transition = Dirichlet(self._dirichlet_beta.clone())
        self._posterior_likelihood = Dirichlet(self._dirichlet_alpha.clone())
        self._posterior_transition = Dirichlet(self._dirichlet_beta.clone())
        self._posterior_policy = torch.softmax(torch.zeros(self._amount_of_policies), dim=-1)

    def initialise_preferences(self, observation=None, value=None, internalise=False):
        if not internalise:
            self._prior_future_observation[observation] = value
        else:
            self._prior_future_observation = torch.softmax(self._prior_future_observation, dim=-1)

    def infer_parameters(self, commit=False):
        if not commit:
            self._posterior_likelihood.concentration = self._prior_likelihood.concentration + self._joint_state_observation.sum(dim=0)
            posterior_transition = self._prior_transition.concentration.clone()
            posterior_transition.index_add_(0, self._action_history, self._joint_state_state)
            self._posterior_transition.concentration = posterior_transition

        else:
            if self._observation_history.size(0) >= self._short_term_memory:
                self._prior_likelihood.concentration *= (1 - self._forgetfulness)
                self._prior_transition.concentration *= (1 - self._forgetfulness)
                self._prior_likelihood.concentration.clamp_(min=1.0)
                self._prior_transition.concentration.clamp_(min=1.0)

                self._prior_likelihood.concentration += self._joint_state_observation[0]
                self._prior_transition.concentration[self._action_history[0]] += self._joint_state_state[0]

    def infer_past(self, convergence_criteria=0.001, upper_limit=1000):
        prev_vfe = float("inf")
        past_observations = F.one_hot(self._observation_history, num_classes=self._amount_of_observations)

        for index in range(upper_limit):
            forward_message = torch.softmax(torch.zeros(self._past_state.size(0), self._amount_of_states), dim=-1)
            backward_message = torch.softmax(torch.zeros(self._past_state.size(0), self._amount_of_states), dim=-1)
            mean_log_likelihood = torch.digamma(self._posterior_likelihood.concentration) - torch.digamma(self._posterior_likelihood.concentration.sum(dim=-1, keepdim=True))
            mean_log_transition = torch.digamma(
                self._posterior_transition.concentration[self._action_history]) - torch.digamma(
                self._posterior_transition.concentration[self._action_history].sum(dim=-1, keepdim=True))
            likelihood = torch.exp(mean_log_likelihood)
            transition = torch.exp(mean_log_transition)

            for t in range(1, forward_message.size(0)):
                forward_message[t] = likelihood[:, self._observation_history[t-1]]*(torch.einsum('ij, i -> j', transition[t-1], forward_message[t-1]))
                backward_message[-(t+1)] = torch.einsum('ij, j -> i', transition[-t], likelihood[:, self._observation_history[-t]]*backward_message[-t])

                forward_message[t] = forward_message[t] / forward_message[t].sum().clamp(min=1e-8)
                backward_message[-(t + 1)] = backward_message[-(t+1)] / backward_message[-(t+1)].sum().clamp(min=1e-8)

            past_state_unnormalised = torch.einsum('ts, ts -> ts', forward_message, backward_message)
            self._past_state = past_state_unnormalised / past_state_unnormalised.sum(dim=-1, keepdim=True)
            self._joint_state_observation = torch.einsum('to, ts -> tso', past_observations,
                                                         self._past_state[1:])
            joint_state_state_unnormalised = torch.einsum('tij, tj -> tij', torch.einsum('ti, tij, jt -> tij', forward_message[:-1], transition, likelihood[:, self._observation_history]), backward_message[1:])
            self._joint_state_state = joint_state_state_unnormalised / joint_state_state_unnormalised.sum(dim=(-1, -2), keepdim=True)
            self.infer_parameters()

            ##### VFE CONVERGENCE CHECK #####

            variational_free_energy_time = (self._past_state[1:] * torch.log(
                self._past_state[1:].clamp(min=1e-8))).sum(dim=-1) - torch.einsum('tso, so -> t',
                                                                                  self._joint_state_observation,
                                                                                  mean_log_likelihood) - torch.einsum(
                'tji, tji -> t',
                self._joint_state_state, mean_log_transition)

            variational_free_energy = variational_free_energy_time.sum(dim=-1) + torch.distributions.kl_divergence(
                self._posterior_likelihood, self._prior_likelihood).sum() + torch.distributions.kl_divergence(
                self._posterior_transition, self._prior_transition).sum()

            if abs(prev_vfe - variational_free_energy) < convergence_criteria:
                break
            prev_vfe = variational_free_energy

        self.infer_parameters(commit=True)

    def infer_future(self):
        mean_log_likelihood = torch.digamma(self._posterior_likelihood.concentration) - torch.digamma(
            self._posterior_likelihood.concentration.sum(dim=-1, keepdim=True))
        mean_log_transition = torch.digamma(
            self._posterior_transition.concentration[self._all_action_sequences]) - torch.digamma(
            self._posterior_transition.concentration[self._all_action_sequences].sum(dim=-1, keepdim=True))
        likelihood = self._posterior_likelihood.mean
        transition = self._posterior_transition.mean[self._all_action_sequences]
        self._future_state[:,0] = torch.softmax(torch.einsum('pij, i -> pj', transition[:,0], self._past_state[-1]), dim=-1)

        for time in range(1, self._planning_horizon):
            self._future_state[:,time] = torch.einsum('pij, pi -> pj', transition[:,time], self._future_state[:,time-1])
        self._future_observation = torch.einsum('so, pts -> pto', likelihood, self._future_state)

        ##### EXPECTED FREE ENERGY #####
        joint_state_observation_policy = torch.einsum('so, pts -> ptso', likelihood, self._future_state)
        joint_state_state_policy = torch.einsum('pti, ptij -> ptij', torch.cat([self._past_state[-1].unsqueeze(0).unsqueeze(0).expand(self._amount_of_policies, 1, -1), self._future_state[:,:-1]], dim=1), transition)

        expected_free_energy_policy_time = ((self._future_state * torch.log(
                self._future_state.clamp(min=1e-8))).sum(dim=-1) + (self._future_observation * torch.log(
                self._future_observation.clamp(min=1e-8))).sum(dim=-1) - (self._future_observation * torch.log(
                self._prior_future_observation.clamp(min=1e-8))).sum(dim=-1) - torch.einsum('ptso, so -> pt',
                                                                                 joint_state_observation_policy,
                                                                                 mean_log_likelihood) - torch.einsum(
                'ptji, ptji -> pt',
                joint_state_state_policy, mean_log_transition))

        expected_free_energy_policy = expected_free_energy_policy_time.sum(dim=-1)

        self._posterior_policy = torch.softmax(torch.log(self._prior_policy.clamp(min=1e-8)) - expected_free_energy_policy, dim=-1)
        self._chosen_policy = torch.multinomial(self._posterior_policy, num_samples=1).item()
        self.action = self._all_action_sequences[int(self._chosen_policy)][0]

    def commit_to_memory(self, observation):
        if self._observation_history.size(0) < self._short_term_memory:
            self._observation_history = torch.cat([self._observation_history, torch.tensor([int(observation)], dtype=torch.long)])
            self._action_history = torch.cat([self._action_history, torch.tensor([int(self.action)], dtype=torch.long)])

            while self._past_state.size(dim=0) < 1 + self._observation_history.size(dim=0):
                self._past_state = torch.vstack(
                    [self._past_state, torch.softmax(torch.zeros(self._amount_of_states), dim=-1)])

        else:
            self._observation_history = torch.roll(self._observation_history, -1)
            self._observation_history[-1] = int(observation)
            self._action_history = torch.roll(self._action_history, -1)
            self._action_history[-1] = int(self.action)



class ActInfAgent:
    def __init__(self, amount_of_states, amount_of_observations, amount_of_actions, short_term_memory, planning_horizon,
                 inference_rate=0.01, maximum_iterations=1000, convergence_criteria=0.001, forgetfulness=0.001, action=None):

        ### Model Structural Values ###

        self._amount_of_states = amount_of_states
        self._amount_of_observations = amount_of_observations
        self._amount_of_actions = amount_of_actions
        self._short_term_memory = short_term_memory
        self._planning_horizon = planning_horizon
        self.action = random.randint(0, amount_of_actions-1)
        self._chosen_policy = 0
        self._amount_of_policies = self._amount_of_actions ** self._planning_horizon
        self._all_action_sequences = torch.cartesian_prod(*[torch.arange(self._amount_of_actions)] * self._planning_horizon)
        self._inference_rate = inference_rate
        self._maximum_iterations = maximum_iterations
        self._convergence_criteria = convergence_criteria
        self._forgetfulness = forgetfulness
        self._ghost_state = torch.zeros(amount_of_states)
        self._hypothetical_future_likelihood = None
        self._hypothetical_future_transition = None

        ### Model Priors ###

        self._prior_past_state_free_logits = torch.zeros(1, self._amount_of_states, requires_grad=False)
        self._prior_future_state_free_logits = torch.zeros(self._amount_of_policies, self._planning_horizon,
                                                          self._amount_of_states,
                                                          requires_grad=False)
        self._prior_future_observation_free_logits = torch.zeros(self._amount_of_observations, requires_grad=False)
        self._prior_policy_free_logits = torch.zeros(self._amount_of_policies, requires_grad=False)
        self._dirichlet_alpha = 1.0 + 1.0 * torch.rand(self._amount_of_states, self._amount_of_observations)
        self._dirichlet_beta = 1.0 + 1.0 * torch.rand(self._amount_of_actions, self._amount_of_states, self._amount_of_states)
        self._prior_likelihood = Dirichlet(self._dirichlet_alpha)
        self._prior_transition = Dirichlet(self._dirichlet_beta)

        ### Variational Posteriors and related ###

        self._observation_history = torch.tensor([])
        self._action_history = torch.tensor([])
        self._posterior_past_state_free_logits = self._prior_past_state_free_logits.detach().clone().requires_grad_(True)
        self._posterior_future_state_free_logits = self._prior_future_state_free_logits.detach().clone().requires_grad_(True)
        self._previous_state_free_logits = None
        self._posterior_policy_free_logits = self._prior_policy_free_logits.detach().clone().requires_grad_(True)
        self._posterior_likelihood = Dirichlet(self._prior_likelihood.concentration.clone())
        self._posterior_transition = Dirichlet(self._prior_transition.concentration.clone())

    def initialise_preferences(self, observation, value):
        self._prior_future_observation_free_logits[observation] = value

    def infer_past(self):
        previous_gfe = float("inf")
        past_observations = F.one_hot(self._observation_history.long(), num_classes=self._amount_of_observations)

        ### BELIEF RESET ###
        # self._posterior_past_state_free_logits = self._prior_past_state_free_logits.detach().clone().requires_grad_(
        #    True)
        #with torch.no_grad():
        #    log_lik = torch.log(self._posterior_likelihood.mean + 1e-8)  # (S, O)
        #    init = log_lik[:, self._observation_history.long()].T  # (T, S)
        #self._posterior_past_state_free_logits = init.clone().requires_grad_(True)
        ####################

        mean_log_likelihood = torch.digamma(self._posterior_likelihood.concentration) - torch.digamma(
            self._posterior_likelihood.concentration.sum(dim=-1, keepdim=True))
        mean_log_transition = torch.digamma(
            self._posterior_transition.concentration[self._action_history.long()]) - torch.digamma(
            self._posterior_transition.concentration[self._action_history.long()].sum(dim=-1, keepdim=True))

        optimizer_past = torch.optim.Adam([self._posterior_past_state_free_logits], lr=self._inference_rate)

        for i in range(self._maximum_iterations):
            optimizer_past.zero_grad()

            ### READABILITY VARIABLES ###
            previous_state_free_logits = torch.cat([self._ghost_state.unsqueeze(0), torch.roll(self._posterior_past_state_free_logits, 1, 0)[1:]], dim=0)
            previous_state = torch.softmax(previous_state_free_logits, dim=-1)
            posterior_past_state = torch.softmax(self._posterior_past_state_free_logits, dim=-1)
            joint_variational_matrix_state_observation = torch.einsum('to, ts -> tso', past_observations,
                                                                      posterior_past_state)
            joint_variational_matrix_state_state = torch.einsum('tj, ti -> tji', previous_state,
                                                                posterior_past_state)  # A temporal mean-field assumption has been made here.
            #############################

            generalised_free_energy_time = (posterior_past_state * torch.log(
                posterior_past_state.clamp(min=1e-8))).sum(dim=-1) - torch.einsum('tso, so -> t',
                                                                                 joint_variational_matrix_state_observation,
                                                                                 mean_log_likelihood) - torch.einsum(
                'tji, tji -> t',
                joint_variational_matrix_state_state, mean_log_transition)

            generalised_free_energy = generalised_free_energy_time.sum(dim=-1) + torch.distributions.kl_divergence(self._posterior_likelihood, self._prior_likelihood).sum() + torch.distributions.kl_divergence(self._posterior_transition, self._prior_transition).sum()

            if abs(previous_gfe - generalised_free_energy) < self._convergence_criteria:
                break
            previous_gfe = generalised_free_energy.item()

            generalised_free_energy.backward()
            optimizer_past.step()

        self.infer_parameters(joint_variational_matrix_state_observation, joint_variational_matrix_state_state)

        return generalised_free_energy


    def infer_future(self):
        ### BELIEF RESET ###
        self._posterior_future_state_free_logits = self._prior_future_state_free_logits.detach().clone().requires_grad_(
            True)
        self._posterior_policy_free_logits = self._prior_policy_free_logits.detach().clone().requires_grad_(
            True)
        ####################

        optimizer_future = torch.optim.Adam([self._posterior_future_state_free_logits, self._posterior_policy_free_logits], lr=self._inference_rate)
        previous_gfe = float("inf")

        for i in range(self._maximum_iterations):
            optimizer_future.zero_grad()

            ### READABILITY VARIABLES ###
            previous_state_free_logits = torch.cat([self._posterior_past_state_free_logits[-1].detach().unsqueeze(0).unsqueeze(0).expand(self._amount_of_policies, 1, -1), torch.roll(self._posterior_future_state_free_logits, 1, 1)[:, 1:]], dim=1)
            previous_state = torch.softmax(previous_state_free_logits, dim=-1)
            posterior_future_state = torch.softmax(self._posterior_future_state_free_logits, dim=-1)
            mean_log_likelihood = torch.digamma(self._posterior_likelihood.concentration) - torch.digamma(
                self._posterior_likelihood.concentration.sum(dim=-1, keepdim=True))
            mean_log_transition = torch.digamma(
                self._posterior_transition.concentration[self._all_action_sequences]) - torch.digamma(
                self._posterior_transition.concentration[self._all_action_sequences].sum(dim=-1, keepdim=True))
            prior_future_observation = torch.softmax(self._prior_future_observation_free_logits, dim=-1)
            posterior_future_observation = torch.einsum('so, pts -> pto', self._posterior_likelihood.mean, posterior_future_state)
            joint_variational_matrix_state_observation = torch.einsum('so, pts -> ptso', self._posterior_likelihood.mean,
                                                                      posterior_future_state)
            joint_variational_matrix_state_state = torch.einsum('ptj, pti -> ptji', previous_state,
                                                                posterior_future_state)  # A temporal mean-field assumption has been made here.
            self.infer_parameters(joint_variational_matrix_state_observation, joint_variational_matrix_state_state, previous_state, posterior_future_observation, hypothetical=True)
            hypothetical_mean_log_likelihood = torch.digamma(self._hypothetical_future_likelihood.concentration) - torch.digamma(
                self._hypothetical_future_likelihood.concentration.sum(dim=-1, keepdim=True))
            hypothetical_mean_log_transition = torch.digamma(
                self._hypothetical_future_transition.concentration) - torch.digamma(
                self._hypothetical_future_transition.concentration.sum(dim=-1, keepdim=True))
            hypothetical_mean_log_transition = hypothetical_mean_log_transition[torch.arange(self._amount_of_policies)[:, None], self._all_action_sequences.long()]
            posterior_policy = torch.softmax(self._posterior_policy_free_logits, dim=0)
            prior_policy = torch.softmax(self._prior_policy_free_logits, dim=0)
            #############################

            generalised_free_energy_policy_time = ((posterior_future_state * torch.log(
                posterior_future_state.clamp(min=1e-8))).sum(dim=-1) + (posterior_future_observation * torch.log(
                posterior_future_observation.clamp(min=1e-8))).sum(dim=-1) - (posterior_future_observation * torch.log(
                prior_future_observation.clamp(min=1e-8))).sum(dim=-1) - torch.einsum('ptso, so -> pt',
                                                                                 joint_variational_matrix_state_observation,
                                                                                 mean_log_likelihood) - torch.einsum(
                'ptji, ptji -> pt',
                joint_variational_matrix_state_state, mean_log_transition))

            expanded_posterior_likelihood = torch.distributions.Dirichlet(
    self._posterior_likelihood.concentration.unsqueeze(0).expand(self._amount_of_policies, -1, -1))
            expanded_posterior_transition = torch.distributions.Dirichlet(self._posterior_transition.concentration.unsqueeze(0).expand(self._amount_of_policies, -1, -1, -1))

            generalised_free_energy_policy = generalised_free_energy_policy_time.sum(
                dim=-1) + torch.distributions.kl_divergence(self._hypothetical_future_likelihood,
                                                            expanded_posterior_likelihood).sum(dim=-1) + torch.distributions.kl_divergence(
                self._hypothetical_future_transition, expanded_posterior_transition).sum(dim=(-1, -2)) - (
                                                     torch.einsum('ptso, pso -> p',
                                                                  joint_variational_matrix_state_observation,
                                                                  hypothetical_mean_log_likelihood - mean_log_likelihood.unsqueeze(0)) + torch.einsum(
                                                 'ptji, ptji -> p',
                                                 joint_variational_matrix_state_state,
                                                 hypothetical_mean_log_transition - mean_log_transition))

            generalised_free_energy = (posterior_policy * (generalised_free_energy_policy + torch.log(posterior_policy.clamp(1e-8)) - torch.log(prior_policy.clamp(1e-8)))).sum(dim=0)

            if abs(previous_gfe - generalised_free_energy) < self._convergence_criteria:
                break
            previous_gfe = generalised_free_energy.item()

            generalised_free_energy.backward()
            optimizer_future.step()

        self._chosen_policy = torch.multinomial(torch.softmax(self._posterior_policy_free_logits, dim=-1), num_samples=1).item()
        self.action = self._all_action_sequences[self._chosen_policy][0]

    def infer_parameters(self, joint_variational_matrix_state_observation=None, joint_variational_matrix_state_state=None, previous_state=None, posterior_future_observation=None, hypothetical=False, commit=False):
        if not hypothetical:
            if not commit:
                likelihood_window_counts = joint_variational_matrix_state_observation.sum(dim=0)
                transition_window_counts = torch.zeros(self._amount_of_actions,
                                                self._amount_of_states, self._amount_of_states,
                                                device=joint_variational_matrix_state_state.device,
                                                dtype=joint_variational_matrix_state_state.dtype)
                transition_window_counts = transition_window_counts.scatter_add(dim=0, index=self._action_history[
                    :, None, None].long().expand_as(joint_variational_matrix_state_state),
                                                                  src=joint_variational_matrix_state_state)

                self._posterior_likelihood.concentration = self._prior_likelihood.concentration.detach().clone() + likelihood_window_counts.detach().clone()
                self._posterior_transition.concentration = self._prior_transition.concentration.detach().clone() + transition_window_counts.detach().clone()

            if commit and not (self._observation_history.size(0) < self._short_term_memory):
                self._prior_likelihood.concentration *= (1 - self._forgetfulness)
                self._prior_transition.concentration *= (1 - self._forgetfulness)
                self._prior_likelihood.concentration.clamp_(min=1.0)
                self._prior_transition.concentration.clamp_(min=1.0)

                successor_state = torch.softmax(self._posterior_past_state_free_logits[0], dim=-1)
                previous_state = torch.softmax(self._ghost_state, dim=-1)
                self._prior_likelihood.concentration[:, self._observation_history[0].long()] += successor_state.detach()
                self._prior_transition.concentration[self._action_history[0].long()] += torch.outer(previous_state, successor_state).detach()

                self._ghost_state = self._posterior_past_state_free_logits[0].detach().clone()

        else:
            self._hypothetical_future_likelihood = Dirichlet(torch.stack([self._posterior_likelihood.concentration.clone()] * self._amount_of_policies))
            self._hypothetical_future_likelihood.concentration += joint_variational_matrix_state_observation.sum(dim=1)

            transition_counts = torch.zeros(self._amount_of_policies, self._amount_of_actions, self._amount_of_states, self._amount_of_states, device=joint_variational_matrix_state_state.device, dtype=joint_variational_matrix_state_state.dtype)
            transition_counts = transition_counts.scatter_add(dim=1, index=self._all_action_sequences[:, :, None, None].expand_as(joint_variational_matrix_state_state), src=joint_variational_matrix_state_state)
            self._hypothetical_future_transition = Dirichlet(self._posterior_transition.concentration.unsqueeze(0) + transition_counts)

    def infer_present(self, rounds=10):
        ### BELIEF RESET ###
        # self._posterior_past_state_free_logits = self._prior_past_state_free_logits.detach().clone().requires_grad_(
        #    True)
        #with torch.no_grad():
        #    log_lik = torch.log(self._posterior_likelihood.mean + 1e-8)  # (S, O)
        #    init = log_lik[:, self._observation_history.long()].T  # (T, S)
        #self._posterior_past_state_free_logits = init.clone().requires_grad_(True)
        ####################

        if self._posterior_past_state_free_logits.size(dim=0) == self._observation_history.size(dim=0):
            self._posterior_past_state_free_logits = torch.cat(
                [torch.roll(self._posterior_past_state_free_logits, -1, 0)[:-1],
                 self._posterior_future_state_free_logits[self._chosen_policy, 0].unsqueeze(0)],
                dim=0).detach().clone().requires_grad_(True)

        else:
            while self._posterior_past_state_free_logits.size(dim=0) < self._observation_history.size(dim=0):
                self._prior_past_state_free_logits = torch.vstack(
                    [self._prior_past_state_free_logits, torch.zeros(self._amount_of_states)])
                self._posterior_past_state_free_logits = torch.cat(
                [self._posterior_past_state_free_logits,
                 self._posterior_future_state_free_logits[self._chosen_policy, 0].unsqueeze(0)],
                dim=0).detach().clone().requires_grad_(True)

        previous_gfe = float("inf")

        for index in range(rounds):
            generalised_free_energy = self.infer_past()

            if abs(previous_gfe - generalised_free_energy) < self._convergence_criteria:
                break
            previous_gfe = generalised_free_energy.item()

        self.infer_parameters(commit=True)

    def commit_to_memory(self, observation):
        if self._observation_history.size(0) < self._short_term_memory:
            self._observation_history = torch.cat([self._observation_history, torch.tensor([observation])])
            self._action_history = torch.cat([self._action_history, torch.tensor([self.action])])

        else:
            self._observation_history = torch.roll(self._observation_history, -1)
            self._observation_history[-1] = observation
            self._action_history = torch.roll(self._action_history, -1)
            self._action_history[-1] = self.action



class Environment1:
    move: int
    light_on: bool

    def __init__(self) -> None:
        self.move = 0
        self.light_on = True
        self.good = 0
        self.bad = 0

    def play_turn(self, should_light_be_on_action: bool) -> int:
        self.light_on = should_light_be_on_action

        print()
        print()

        if self.good + self.bad > 50:
            print()
            print("NEW BATCH!")
            print()
            self.good = 0
            self.bad = 0

        current_move = self.move % 4
        if self.light_on:
            print("Clara is on move", current_move, "and lights are on.")

        else:
            print("Clara is on move", current_move, "and lights are off.")

        if current_move == 0:
            is_moving_this_turn = bool(random.getrandbits(1))

            if is_moving_this_turn:
                self.move += 1
                if not self.light_on:
                    self.good += 1
                    print("Door moves.")
                    print("Performance: ", round(100 * self.good / (self.good + self.bad), 1), "%")
                    return 2
                else:
                    # The model doesnt know that the door moved in this case?
                    self.bad += 1
                    print("Door moves. Clara is unhappy :(")
                    print("Performance: ", round(100 * self.good / (self.good + self.bad), 1), "%")
                    return 1
            else:
                if self.good + self.bad == 0:
                    print("Performance: None")
                else:
                    print("Performance: ", round(100 * self.good / (self.good + self.bad), 1), "%")
                return 3

        if current_move == 3:
            self.move += 1
            if not self.light_on:
                self.good += 1
                print("Clara is happy :)")
                print("Performance: ", round(100 * self.good / (self.good + self.bad), 1), "%")
                return 0
            else:
                self.bad += 1
                print("Clara is unhappy :(")
                print("Performance: ", round(100 * self.good / (self.good + self.bad), 1), "%")
                return 1

        else:
            self.move += 1
            if self.light_on:
                self.good += 1
                print("Clara is happy :)")
                print("Performance: ", round(100 * self.good / (self.good + self.bad), 1), "%")
                return 0
            else:
                self.bad += 1
                print("Clara is unhappy :(")
                print("Performance: ", round(100 * self.good / (self.good + self.bad), 1), "%")
                return 1



class Environment2:
    def __init__(self):
        self._state = 0
        self._good = 0
        self._bad = 0

    def play_turn(self, action):
        self._state = (self._state + action) % 2

        if self._good + self._bad > 200:
            print()
            print("NEW BATCH!")
            print()
            self._good = 0
            self._bad = 0

        if self._state == 0:
            self._good += 1
            print("Good", round(100*self._good/(self._good + self._bad), 1), "%")
            return 0
        else:
            self._bad += 1
            print("Bad", round(100*self._good/(self._good + self._bad), 1), "%")
            return 1



class Environment3:
    def __init__(self, board_size, cell_size=60):
        self._win = 0
        self._loss = 0
        self._board_size = board_size
        self._cell = cell_size
        self._coin   = torch.tensor([random.randint(0, board_size - 1),
                                     random.randint(0, board_size - 1)])
        self._player = torch.tensor([random.randint(0, board_size - 1),
                                     random.randint(0, board_size - 1)])
        self._random = torch.tensor([random.randint(0, board_size - 1),
                                     random.randint(0, board_size - 1)])
        pygame.init()
        self._screen = pygame.display.set_mode((board_size * cell_size,
                                                board_size * cell_size))

    def _render(self):
        pygame.event.pump()
        self._screen.fill((0, 0, 0))
        pygame.draw.rect(self._screen, (240, 240, 60),
                         (int(self._coin[0])   * self._cell, int(self._coin[1])   * self._cell, self._cell, self._cell))
        #pygame.draw.rect(self._screen, (0, 0, 200),
                 #        (int(self._random[0]) * self._cell, int(self._random[1]) * self._cell, self._cell, self._cell))
        pygame.draw.rect(self._screen, (200, 0, 0),
                         (int(self._player[0]) * self._cell, int(self._player[1]) * self._cell, self._cell, self._cell))
        pygame.display.flip()

    def play_turn(self, action):
        action = int(action)

        while (int(self._player[0]) == int(self._coin[0])) and (int(self._player[1]) == int(self._coin[1])): #or (int(self._random[0]) == int(self._coin[0]) and
               #int(self._random[1]) == int(self._coin[1])):
            self._coin = torch.tensor([random.randint(0, self._board_size - 1),
                                       random.randint(0, self._board_size - 1)])

        if action == 0:
            if (self._player[0] + 1) < self._board_size:
                self._player[0] += 1
        elif action == 1:
            if (self._player[1] + 1) < self._board_size:
                self._player[1] += 1
        elif action == 2:
            if (self._player[0] - 1) > -1:
                self._player[0] -= 1
        else:
            if (self._player[1] - 1) > -1:
                self._player[1] -= 1

        #action_rand = random.randint(0, 3)

        #if action_rand == 0:
        #    self._random[0] = (self._random[0] + 1) % self._board_size
        #elif action == 1:
        #    self._random[1] = (self._random[1] + 1) % self._board_size
        #elif action == 2:
        #    self._random[0] = (self._random[0] - 1) % self._board_size
        #else:
        #    self._random[1] = (self._random[1] - 1) % self._board_size

        if (int(self._player[0]) == int(self._coin[0]) and int(self._player[1]) == int(self._coin[1])):
            self._win += 1
            print("AGENT SCORE: ", self._win)
            #print("RANDOM SCORE: ", self._loss)
            print()

        #if (int(self._random[0]) == int(self._coin[0]) and int(self._random[1]) == int(self._coin[1])):
        #    self._loss += 1
        #    print("AGENT SCORE: ", self._win)
        #    print("RANDOM SCORE: ", self._loss)
        #    print()

        self._render()

        return abs(self._player - self._coin).sum(dim=-1)