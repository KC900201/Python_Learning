
# Create a Mario class represent agent in the game.
# Agent should be able to -> Act, Remember, Learn

# Shell class
class Mario:
    def __init__(self):
        pass

    def act(self, state):
        """Given a state, choose an epsilon-greedy action"""
        pass

    def cache(self, experience):
        """Add the experience to memory"""
        pass

    def recall(self):
        """Sample experiences from memory"""
        pass

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        pass

# Contine Act agent (4/30/2021)