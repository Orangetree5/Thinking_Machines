import torch
import torch.nn.functional as F
import random
from torch.distributions import Dirichlet

from ACTIVEINFERENCE import posterior_state_free_logits, ghost_state, amount_of_observations


class ActInfAgent:
    def __init__(self, amount_of_states, amount_of_observations, amount_of_actions, short_term_memory, planning_horizon,
                 inference_rate=0.01, maximum_iterations=1000, convergence_criteria=0.001, action=None):

        ### Model Structural Values ###

        self._amount_of_states = amount_of_states
        self._amount_of_observations = amount_of_observations
        self._amount_of_actions = amount_of_actions
        self._short_term_memory = short_term_memory
        self._planning_horizon = planning_horizon
        self.action = random.randint(0, amount_of_actions-1)
        self._chosen_policy = None
        self._amount_of_policies = self._amount_of_actions ** self._planning_horizon
        self._all_action_sequences = torch.cartesian_prod(*[torch.arange(self._amount_of_actions)] * self._planning_horizon)
        self._inference_rate = inference_rate
        self._maximum_iterations = maximum_iterations
        self._convergence_criteria = convergence_criteria

        ### Model Priors ###

        self._prior_past_state_free_logits = torch.zeros(1, self._amount_of_states, requires_grad=False)
        self._prior_future_state_free_logits = torch.zeros(self._amount_of_policies, self._planning_horizon,
                                                          self._amount_of_states,
                                                          requires_grad=False)
        self._prior_future_observation_free_logits = torch.zeros(self._amount_of_observations, requires_grad=False)
        self._prior_policy_free_logits = torch.zeros(self._amount_of_policies, requires_grad=False)
        self._dirichlet_alpha = torch.ones(self._amount_of_observations, self._amount_of_states)
        self._dirichlet_beta = torch.ones(self._amount_of_actions, self._amount_of_states, self._amount_of_states)
        self._prior_likelihood = Dirichlet(self._dirichlet_alpha)
        self._prior_transition = Dirichlet(self._dirichlet_beta)

        ### Variational Posteriors and related ###

        self._observation_history = torch.tensor([])
        self._action_history = torch.tensor([])
        self._posterior_past_state_free_logits = self._prior_past_state_free_logits.detach().clone().requires_grad_(True)
        self._posterior_future_state_free_logits = self._prior_future_state_free_logits.detach().clone().requires_grad_(True)
        self._posterior_future_observation_free_logits = None
        self._posterior_policy_free_logits = self._prior_policy_free_logits.detach().clone().requires_grad_(True)
        self._posterior_likelihood = Dirichlet(self._dirichlet_alpha.clone())
        self._posterior_transition = Dirichlet(self._dirichlet_beta.clone())

    def initialise_preferences(self, observation, value):
        self._prior_future_observation_free_logits[observation] = value

    def infer_past(self):
        while self._posterior_past_state_free_logits.size(dim=0) < self._observation_history.size(dim=0):
            self._prior_past_state_free_logits = torch.vstack(
                [torch.zeros(self._amount_of_states), self._prior_past_state_free_logits])
            self._posterior_past_state_free_logits = torch.vstack(
                [torch.zeros(self._amount_of_states), self._posterior_past_state_free_logits])

        optimizer_past = torch.optim.Adam([self._posterior_past_state_free_logits], lr=self._inference_rate)
        previous_gfe = float("inf")
        ghost_state = self._posterior_past_state_free_logits[0].detach().clone()
        past_observations = F.one_hot(self._observation_history, num_classes=self._amount_of_observations)

        ### BELIEF RESET ###
        self._posterior_past_state_free_logits = self._prior_past_state_free_logits.detach().clone().requires_grad_(True)
        ####################

        for i in range(self._maximum_iterations):
            optimizer_past.zero_grad()

            ### READABILITY VARIABLES ###
            previous_state_free_logits = torch.roll(self._posterior_past_state_free_logits, 1, 0)
            previous_state_free_logits[0] = ghost_state
            previous_state = torch.softmax(previous_state_free_logits, dim=-1)
            posterior_past_state = torch.softmax(self._posterior_past_state_free_logits, dim=-1)
            joint_variational_matrix_observation_state = torch.einsum('to, ts -> tos', past_observations,
                                                                      posterior_past_state)
            joint_variational_matrix_state_state = torch.einsum('ts, ts -> tss', previous_state,
                                                                posterior_past_state)  # A temporal mean-field assumption has been made here.
            mean_log_likelihood = torch.digamma(self._posterior_likelihood.concentration) - torch.digamma(
                self._posterior_likelihood.concentration.sum(dim=-1, keepdim=True))
            mean_log_transition = torch.digamma(
                self._posterior_transition.concentration[self._action_history]) - torch.digamma(
                self._posterior_transition.concentration[self._action_history].sum(dim=-1, keepdim=True))
            #############################

            generalised_free_energy_time = (posterior_past_state * torch.log(
                posterior_past_state.clamp(min=1e-8))).sum(dim=-1) - torch.einsum('tso, os -> t',
                                                                                 joint_variational_matrix_observation_state,
                                                                                 mean_log_likelihood) - torch.einsum(
                'tkj, tjk -> t',
                joint_variational_matrix_state_state, mean_log_transition)

            generalised_free_energy = generalised_free_energy_time.sum(dim=-1)

            if abs(previous_gfe - generalised_free_energy) < self._convergence_criteria:
                break
            previous_gfe = generalised_free_energy.item()

            generalised_free_energy.backward()
            optimizer_past.step()

    def infer_future(self):
        optimizer_future = torch.optim.Adam([self._posterior_future_state_free_logits, self._posterior_policy_free_logits], lr=self._inference_rate)
        previous_gfe = float("inf")

        ### BELIEF RESET ###
        self._posterior_future_state_free_logits = self._prior_future_state_free_logits.detach().clone().requires_grad_(
            True)
        ####################

        for i in range(self._maximum_iterations):
            optimizer_future.zero_grad()

            ### READABILITY VARIABLES ###
            previous_state_free_logits = torch.roll(self._posterior_past_state_free_logits, 1, 0)
            previous_state_free_logits[0] = ghost_state
            previous_state = torch.softmax(previous_state_free_logits, dim=-1)
            posterior_future_state = torch.softmax(self._posterior_future_state_free_logits, dim=-1)
            mean_log_likelihood = torch.digamma(self._posterior_likelihood.concentration) - torch.digamma(
                self._posterior_likelihood.concentration.sum(dim=-1, keepdim=True))
            mean_log_transition = torch.digamma(
                self._posterior_transition.concentration[self._all_action_sequences]) - torch.digamma(
                self._posterior_transition.concentration[self._all_action_sequences].sum(dim=-1, keepdim=True))
            posterior_future_observation = torch.einsum('os, pts -> pto', self._posterior_likelihood.mean, posterior_future_state)
            joint_variational_matrix_observation_state = torch.einsum('to, ts -> tos', posterior_future_observation,
                                                                      posterior_future_state)
            joint_variational_matrix_state_state = torch.einsum('ts, ts -> tss', previous_state,
                                                                posterior_future_state)  # A temporal mean-field assumption has been made here.
            #############################

            generalised_free_energy = None

            if abs(previous_gfe - generalised_free_energy) < self._convergence_criteria:
                break
            previous_gfe = generalised_free_energy.item()

            generalised_free_energy.backward()
            optimizer_future.step()

        self._chosen_policy = torch.multinomial(torch.softmax(self._posterior_policy_free_logits, dim=-1), num_samples=1).item()
        self.action = self._all_action_sequences[self._chosen_policy][0]

    def infer_parameters(self):
        self._posterior_likelihood.concentration[:, self._observation_history[-1]] += torch.softmax(self._posterior_past_state_free_logits[-1, :], dim=-1)
        successor_state = torch.softmax(self._posterior_past_state_free_logits[-1], dim=-1)
        previous_state = torch.softmax(self._posterior_past_state_free_logits[-2], dim=-1)
        self._posterior_transition.concentration[self._action_history[-1]] += torch.outer(previous_state, successor_state)

    def commit_to_memory(self, observation):
        if self._observation_history < self._short_term_memory:
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

    def play_turn(self, should_light_be_on_action: bool) -> int:
        self.light_on = should_light_be_on_action

        print()
        print()

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
                    print("Door moves.")
                    return 2
                else:
                    # The model doesnt know that the door moved in this case?
                    print("Door moves. Clara is unhappy :(")
                    return 1
            else:
                return 3

        if current_move == 3:
            self.move += 1
            if not self.light_on:
                print("Clara is happy :)")
                return 0
            else:
                print("Clara is unhappy :(")
                return 1

        else:
            self.move += 1
            if self.light_on:
                print("Clara is happy :)")
                return 0
            else:
                print("Clara is unhappy :(")
                return 1