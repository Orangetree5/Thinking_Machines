from random import randint

import torch
import torch.nn.functional as F
import random

inference_rate = 0.1
learning_rate = 0.01
weight_decay = 0
maximum_iterations = 1000
convergence_criteria = 0.001
amount_of_states = 10
amount_of_observations = 4
amount_of_actions = 2
planning_horizon = 3
short_term_memory = 500
total_time = short_term_memory + planning_horizon
amount_of_policies = amount_of_actions ** planning_horizon
amount_of_actions_before_next_think = 1

# Generative Model
prior_over_state_free_logits = torch.ones(amount_of_policies, total_time, amount_of_states, requires_grad=False)
prior_over_future_observation_free_logits = torch.zeros(amount_of_observations, requires_grad=False)
prior_over_future_observation_free_logits[0] = 3.
prior_over_future_observation_free_logits[1] = -3.
prior_over_policy_free_logits = torch.ones(amount_of_policies, requires_grad=False)
likelihood_free_logits = torch.randn(amount_of_states, amount_of_observations, requires_grad=False)
transition_free_logits = torch.randn(amount_of_actions, amount_of_states, amount_of_states, requires_grad=False)

prior_over_state = torch.softmax(prior_over_state_free_logits, dim=2)
prior_over_future_observation = torch.softmax(prior_over_future_observation_free_logits, dim=0)
prior_over_observation = torch.cat([torch.ones(short_term_memory, amount_of_observations), prior_over_future_observation.expand(planning_horizon, -1)], dim=0)
prior_over_policy = torch.softmax(prior_over_policy_free_logits, dim=0)
likelihood = torch.softmax(likelihood_free_logits, dim=1)
transition = torch.softmax(transition_free_logits, dim=2)

actions = torch.arange(amount_of_actions)
all_action_sequences = torch.cartesian_prod(*[actions] * planning_horizon)
transition_policy = transition[all_action_sequences]

posterior_state_free_logits = prior_over_state_free_logits.detach().clone().requires_grad_(True)
posterior_state = torch.softmax(posterior_state_free_logits, dim=2)

###MEMORY
action_history = torch.zeros(short_term_memory, requires_grad=False)
observation_history = torch.zeros(short_term_memory, requires_grad=False)

posterior_policy_free_logits = prior_over_policy_free_logits.detach().clone().requires_grad_(True)

chosen_policy = 0

world_state = 0


def environment1(action):
    global world_state

    world_state = (world_state + action) % 2

    if world_state == 0:
        return 0
    else:
        return random.choice([1, 2])


move, light = 0, 0


def environment3(action):
    global world_state, move, light
    state_collection = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])

    light = light + action.long()
    world_state = state_collection[move % 4][light % 2]

    print("")
    print("")

    if (light % 2) == 0:
        print("Clara is on move", move % 4, "and lights are off.")
    else:
        print("Clara is on move", move % 4, "and lights are on.")

    if world_state in (0, 1):
        if random.randint(0, 2) == 0:
            move += 1
            if world_state == 0:
                print("Door moves.")
                return 2
            else:
                print("Door moves. Clara is unhappy :(")
                return 1

        else:
            return 3
    if world_state in (2, 3):
        move += 1
        if world_state == 2:
            print("Clara is unhappy :(")
            return 1
        else:
            print("Clara is happy :)")
            return 0
    if world_state in (4, 5):
        move += 1
        if world_state == 4:
            print("Clara is unhappy :(")
            return 1
        else:
            print("Clara is happy :)")
            return 0
    else:
        move += 1
        if world_state == 6:
            print("Clara is happy :)")
            return 0
        else:
            print("Clara is unhappy :(")
            return 1



def safe_log(x, eps=1e-12):
    return torch.log(torch.clamp(x, min=eps))



for initial_memory in range(short_term_memory):
    action_history = torch.roll(action_history, -1)
    action_history[-1] = random.randint(0, amount_of_actions-1)

    observation_history = torch.roll(observation_history, -1)
    observation_history[-1] = environment3(action_history[-1])

delayed_obs = observation_history
delayed_act = action_history

model_optimizer = torch.optim.Adam([likelihood_free_logits, transition_free_logits, prior_over_policy_free_logits], lr=learning_rate, weight_decay=weight_decay)

while True:
    ##### THIS VARIABLE IS DEFINED ONLY ONCE IMMIDIDIATLY AFTER AN ACTION IS TAKEN
    ghost_state = posterior_state[chosen_policy][amount_of_actions_before_next_think - 1].detach()
    ##### THEN A NEW GENERATION OF STATE AND OBSERVATION TENSORS ARE GENERATED FOR THE NEW INFERENCE

    # Variational posteriors
    #posterior_state_free_logits = torch.cat([posterior_state_free_logits[:, 1:], torch.ones((amount_of_policies, 1, amount_of_states), device=posterior_state_free_logits.device)], dim=1)
    #posterior_state_free_logits = posterior_state_free_logits.detach().requires_grad_(True)
    #rolled_posterior = torch.cat([posterior_state_free_logits[:, 1:].detach(), prior_over_state_free_logits[:, -1:].detach()], dim=1)
    #posterior_state_free_logits = rolled_posterior.clone().requires_grad_(True)
    posterior_state_free_logits = prior_over_state_free_logits.detach().clone().requires_grad_(True)
    posterior_future_observation_free_logits = torch.randn(amount_of_policies, planning_horizon, amount_of_observations,
                                                           requires_grad=False)

    ### with memory
    posterior_observation_free_logits = torch.cat(
        [torch.broadcast_to(F.one_hot(observation_history.long(), num_classes=amount_of_observations),
                            (posterior_future_observation_free_logits.size(0),
                             *F.one_hot(observation_history.long(), num_classes=amount_of_observations).shape)),
         posterior_future_observation_free_logits], dim=1)
    posterior_policy_free_logits = prior_over_policy_free_logits.detach().clone().requires_grad_(True)

    transition_history = transition[action_history.long()]
    all_transitions = torch.cat(
        [torch.broadcast_to(transition_history.unsqueeze(0), (transition_policy.size(0), *transition_history.shape)),
         transition_policy], dim=1)


    ### Preperation for inference
    posterior_state = torch.softmax(posterior_state_free_logits, dim=2)
    previous_state = \
        torch.cat([ghost_state[None, None, :].expand(posterior_state.size(0), 1, -1), posterior_state], dim=1)[
            :, :-1]
    posterior_observation_free_logits = torch.cat(
        [torch.broadcast_to(F.one_hot(observation_history.long(), num_classes=amount_of_observations),
                            (posterior_future_observation_free_logits.size(0),
                             *F.one_hot(observation_history.long(), num_classes=amount_of_observations).shape)),
         posterior_future_observation_free_logits], dim=1)
    posterior_policy = torch.softmax(posterior_policy_free_logits, dim=0)

    gradient_magnitude = 10

    kronecker_delta_likelihood = F.one_hot(observation_history.long(), num_classes=amount_of_observations).unsqueeze(1).expand(-1, amount_of_states, -1)
    future_likelihood = likelihood.unsqueeze(0).expand(planning_horizon, -1, -1)
    variational_likelihood = torch.cat([kronecker_delta_likelihood, future_likelihood], dim=0)

    posterior_observation = torch.einsum('tso, pts -> pto', variational_likelihood, posterior_state)

    optimizer = torch.optim.Adam([posterior_state_free_logits, posterior_policy_free_logits], lr=inference_rate, weight_decay=weight_decay)
    previous_gfe = float('inf')

    ### FAST INFERENCE
    for i in range(maximum_iterations):
        optimizer.zero_grad()

        Generalised_Free_Energy_policy_time = (-(posterior_observation*safe_log(prior_over_observation)).sum(dim=2)
                                                   +(posterior_observation*safe_log(posterior_observation)).sum(dim=2)
                                                   +(posterior_state*posterior_state_free_logits).sum(dim=2)
                                                   -torch.einsum('...so, ...s, ...so -> ...', variational_likelihood, posterior_state, safe_log(likelihood))
                                                   -(posterior_state*safe_log(torch.einsum('btij, bti -> btj', all_transitions, previous_state))).sum(dim=2)
                                                   -torch.logsumexp(posterior_state_free_logits, dim=2)
                                                   #+safe_log(torch.einsum('...i, ...ki, ...jk, ...j -> ...', prior_over_observation, likelihood, all_transitions, previous_state))
                                                    )

        Generalised_Free_Energy_policy = Generalised_Free_Energy_policy_time.sum(dim=1)
        Generalised_Free_Energy = (posterior_policy * (Generalised_Free_Energy_policy + safe_log(posterior_policy) - safe_log(prior_over_policy))).sum(dim=0)

        #print(Generalised_Free_Energy)

        if abs(previous_gfe - Generalised_Free_Energy) < convergence_criteria:
            break
        previous_gfe = Generalised_Free_Energy.item()

        Generalised_Free_Energy.backward()
        optimizer.step()

        with torch.no_grad():
            posterior_state_free_logits.sub_(posterior_state_free_logits.mean(dim=2, keepdim=True))
            posterior_policy_free_logits.sub_(posterior_policy_free_logits.mean(dim=0, keepdim=True))

        posterior_state = torch.softmax(posterior_state_free_logits, dim=2)
        previous_state = \
            torch.cat([ghost_state[None, None, :].expand(posterior_state.size(0), 1, -1), posterior_state], dim=1)[
                :, :-1]
        #posterior_observation_free_logits = torch.cat(
         #   [torch.broadcast_to(F.one_hot(observation_history.long(), num_classes=amount_of_observations),
          #                      (posterior_future_observation_free_logits.size(0),
           #                      *F.one_hot(observation_history.long(), num_classes=amount_of_observations).shape)),
            # posterior_future_observation_free_logits], dim=1)
        #posterior_observation = torch.softmax(posterior_observation_free_logits, dim=2)
        posterior_observation = torch.einsum('tso, pts -> pto', variational_likelihood, posterior_state)
        posterior_policy = torch.softmax(posterior_policy_free_logits, dim=0)

    # ==========================================
    # SLOW INFERENCE: Learning
    # ==========================================

    posterior_state_free_logits.requires_grad_(False)
    #posterior_future_observation_free_logits.requires_grad_(False)
    posterior_policy_free_logits.requires_grad_(False)
    posterior_state_free_logits = posterior_state_free_logits.detach()
    #posterior_future_observation_free_logits = posterior_future_observation_free_logits.detach()
    posterior_policy_free_logits = posterior_policy_free_logits.detach()
    posterior_state = torch.softmax(posterior_state_free_logits, dim=2)
    previous_state = \
        torch.cat([ghost_state[None, None, :].expand(posterior_state.size(0), 1, -1), posterior_state], dim=1)[
            :, :-1]
    #posterior_observation_free_logits = torch.cat(
     #   [torch.broadcast_to(F.one_hot(observation_history.long(), num_classes=amount_of_observations),
      #                      (posterior_future_observation_free_logits.size(0),
       #                      *F.one_hot(observation_history.long(), num_classes=amount_of_observations).shape)),
        # posterior_future_observation_free_logits], dim=1)
    #posterior_observation = torch.softmax(posterior_observation_free_logits, dim=2)
    posterior_policy = torch.softmax(posterior_policy_free_logits, dim=0)

    prior_over_state_free_logits.requires_grad_(False)
    prior_over_policy_free_logits.requires_grad_(False)
    likelihood_free_logits.requires_grad_(True)
    transition_free_logits.requires_grad_(True)

    prior_over_state = torch.softmax(prior_over_state_free_logits, dim=2)
    prior_over_policy = torch.softmax(prior_over_policy_free_logits, dim=0)
    likelihood = torch.softmax(likelihood_free_logits, dim=1)
    transition = torch.softmax(transition_free_logits, dim=2)

    #model_optimizer = torch.optim.Adam([likelihood_free_logits, transition_free_logits], lr=learning_rate)

    for index in range(10):
        transition_policy = transition[all_action_sequences]
        transition_history = transition[action_history.long()]
        all_transitions = torch.cat(
            [torch.broadcast_to(transition_history.unsqueeze(0),
                                (transition_policy.size(0), *transition_history.shape)),
             transition_policy], dim=1)
        future_likelihood = likelihood.unsqueeze(0).expand(planning_horizon, -1, -1)
        variational_likelihood = torch.cat([kronecker_delta_likelihood, future_likelihood], dim=0)
        posterior_observation = torch.einsum('tso, pts -> pto', variational_likelihood, posterior_state)

        model_optimizer.zero_grad()

        Generalised_Free_Energy_policy_time = (-(posterior_observation * safe_log(prior_over_observation)).sum(dim=2)
                                               + (posterior_observation * safe_log(posterior_observation)).sum(dim=2)
                                               + (posterior_state * posterior_state_free_logits).sum(dim=2)
                                               - torch.einsum('...so, ...s, ...so -> ...', variational_likelihood, posterior_state,
                                                              safe_log(likelihood))
                                               - (posterior_state * safe_log(
                    torch.einsum('btij, bti -> btj', all_transitions, previous_state))).sum(dim=2)
                                               - torch.logsumexp(posterior_state_free_logits, dim=2)
                                               #+ safe_log(torch.einsum('...i, ...ki, ...jk, ...j -> ...', prior_over_observation, likelihood, all_transitions, previous_state))
                    )
        Generalised_Free_Energy_policy = Generalised_Free_Energy_policy_time[:, :short_term_memory].sum(dim=1)
        Generalised_Free_Energy = (posterior_policy * (
                    Generalised_Free_Energy_policy + safe_log(posterior_policy) - safe_log(prior_over_policy))).sum(dim=0)

        Generalised_Free_Energy.backward()
        model_optimizer.step()

        with torch.no_grad():
            likelihood_free_logits.sub_(likelihood_free_logits.mean(dim=1, keepdim=True))
            transition_free_logits.sub_(transition_free_logits.mean(dim=2, keepdim=True))

        #prior_over_state = torch.softmax(prior_over_state_free_logits, dim=2)
        prior_over_policy = torch.softmax(prior_over_policy_free_logits, dim=0)
        likelihood = torch.softmax(likelihood_free_logits, dim=1)
        transition = torch.softmax(transition_free_logits, dim=2)
        kronecker_delta_likelihood = F.one_hot(observation_history.long(),
                                               num_classes=amount_of_observations).unsqueeze(1).expand(-1,
                                                                                                       amount_of_states,
                                                                                                       -1)

    prior_over_state_free_logits.requires_grad_(False)
    prior_over_policy_free_logits.requires_grad_(False)
    likelihood_free_logits.requires_grad_(False)
    transition_free_logits.requires_grad_(False)

    prior_over_state = torch.softmax(prior_over_state_free_logits, dim=2)
    prior_over_policy = torch.softmax(prior_over_policy_free_logits, dim=0)
    likelihood = torch.softmax(likelihood_free_logits, dim=1)
    transition = torch.softmax(transition_free_logits, dim=2)

    transition_policy = transition[all_action_sequences]
    transition_history = transition[action_history.long()]
    all_transitions = torch.cat(
        [torch.broadcast_to(transition_history.unsqueeze(0), (transition_policy.size(0), *transition_history.shape)),
         transition_policy], dim=1)

    posterior_state_free_logits.requires_grad_(True)
    #posterior_future_observation_free_logits.requires_grad_(True)
    posterior_policy_free_logits.requires_grad_(True)

    chosen_policy = torch.multinomial(posterior_policy, num_samples=1).item()

    for amount_of_moves in range(amount_of_actions_before_next_think):
        action_history = torch.roll(action_history, -1)
        action_history[-1] = all_action_sequences[chosen_policy][amount_of_moves]
        # action_history[-1] = random.randint(0, 1)

        print("")
        print(delayed_obs, "<--- Past observations")
        print(delayed_act, all_action_sequences[chosen_policy], "<--- Actions: past & future")
        print("")
        print("")
        print("")

        observation_history = torch.roll(observation_history, -1)
        observation_history[-1] = environment3(action_history[-1])

        delayed_obs = observation_history
        delayed_act = action_history



    ### 1. Major change: the partition/normalisation term of the generalised free energy was simply removed.
    ### 2. The prior over observations actually became the prior over FUTURE observations as it should have been,
    ### with the prior over past observations being a flat distribution.
    ### 3. The action taken and the new observation are both inserted into the memory only AFTER slow learning.
    ### so that the state, observation and action time windows are aligned.
