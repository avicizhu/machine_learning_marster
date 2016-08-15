import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    Q = defaultdict(dict)
    epsilon = 1
    alpha = 1
    penalty = 0
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.total_reward = 0
        self.gamma = 0.8
        self.shortest_path = 0
        self.init_deadline = 0
        LearningAgent.penalty = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        #decay alpha and epsilon
        LearningAgent.epsilon /= 1.06
        LearningAgent.alpha -= 0.01
        LearningAgent.penalty = 0
        self.total_reward = 0
        self.init_deadline = self.env.agent_states[self]['deadline']

        #get shortest_path as minimum possible time to reach the destination
        x1, y1 = self.env.agent_states[self]['location']
        x2, y2 = self.env.agent_states[self]['destination']
        min_x = min(abs(x1-x2), self.env.bounds[2] - abs(x1-x2))
        min_y = min(abs(y1-y2), self.env.bounds[3] - abs(y1-y2))
        self.shortest_path = float(min_x + min_y)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        state0 = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])

        # TODO: Select action according to your policy
        #action = random.choice(Environment.valid_actions) # random action

        # take best action or exploration
        p_key = hash(state0)
        Qs = [LearningAgent.Q[p_key].get(x, 0) for x in Environment.valid_actions]
        action = random.choice(Environment.valid_actions) if random.uniform(0,1) <= LearningAgent.epsilon else Environment.valid_actions[Qs.index(max(Qs))]

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward
        if reward == -1:
            LearningAgent.penalty += 1

        # TODO: Learn policy based on state, action, reward
        # sense again to get next state
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])

        # get the max Q' value
        n_key = hash(self.state)
        Qs = [LearningAgent.Q[n_key].get(x, 0) for x in Environment.valid_actions]
        max_action = Environment.valid_actions[Qs.index(max(Qs))]

        #update Q-table
        a = LearningAgent.alpha
        LearningAgent.Q[p_key][action] = (1 - a) * LearningAgent.Q[p_key].get(action, 0) + a * (reward + self.gamma * LearningAgent.Q[n_key].get(max_action, 0))

        # generate output information
        if (deadline == 0) or (self.next_waypoint is None):
            time_used = float(self.init_deadline - deadline + 1)
            ratio = self.shortest_path / time_used if self.next_waypoint is None else 0
            res.append((self.total_reward, 1 if self.next_waypoint is None else 0, LearningAgent.penalty, ratio))


        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    res = []
    run()
    print res
