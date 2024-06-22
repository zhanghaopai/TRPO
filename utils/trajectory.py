class Trajectory():
    def __init__(self):
        self.state=[]
        self.action=[]
        self.next_state=[]
        self.reward=[]
        self.done=[]

    def add(self, state, action, reward, done, next_state):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        self.next_state.append(next_state)