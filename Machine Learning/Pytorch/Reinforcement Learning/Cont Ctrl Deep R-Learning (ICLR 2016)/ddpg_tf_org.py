# Youtube reference for paper implementation - https://www.youtube.com/watch?v=GJJc1t0rtSU&ab_channel=freeCodeCamp.org

# Need a replay buffer class
# Need a class for a  target Q network
# Use batch-normalization (normalizing inputs)
# policy is deterministic, how to handle explorer exploit dilemma?
# deterministic policy ~ outputs the actual action instead of a probability
# will need a way to bound the actions to the env limits
# We have two actor and two critic networks, a target for each.
# Updates are soft, according to theta_prime = tau*theta + (1-tau)*theta_prime, with tau << 1
# The target actor is just the evaluation actor plus some noise process
# they used Ornstein Uhlenbeck (look up later!) -> need a class for the noise
