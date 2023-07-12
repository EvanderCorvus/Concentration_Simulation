import numpy as np

class BoxEnvironment1():
    def __init__(self, space):
        self.space = space
    
    def init_env(self, agent_batch_size, state_dim, goal, c0, random_start = False):
        self.agent_batch_size = agent_batch_size
        self.goal = goal
        if not random_start:
            self.state = np.zeros((agent_batch_size,state_dim))
            self.state[:,0] = -0.5*np.ones(agent_batch_size)
        else:
            raise Exception('test')
            self.state = np.random.uniform(-1, 1, (agent_batch_size, state_dim))
        self.state[:,-1] = compute_concentration(self.state[:,0:2], self.goal.center, c0)
        
    
    def step(self, action, c0, dt, characteristic_length = 1, test = False):
        x, y = self.state[:,0], self.state[:,1]
        theta = action[:,0]
        #thermal noise
        noise = np.random.normal(np.zeros(self.agent_batch_size), np.ones(self.agent_batch_size))
        theta = theta + np.sqrt(dt)*characteristic_length*noise

        e_x = np.cos(theta)
        v_x = e_x 
        x_new = x + v_x*dt

        e_y = np.sin(theta)
        v_y = e_y 
        y_new = y + v_y*dt
        inside_space = self.space.contains(np.array([x_new, y_new]).T)
        concentration = compute_concentration(np.array([x_new, y_new]).T, self.goal.center, c0)
        
        self.state[:,0][inside_space] = x_new[inside_space]
        self.state[:,1][inside_space] = y_new[inside_space]
        self.state[:,2] = concentration 

        # Compute reward
        if not test:
            reward = self.reward(dt, np.array(inside_space).astype(int))
            return reward
    
    def reward(self, dt, inside_space):
        # Compute reward
        not_inside_space = np.logical_not(inside_space)
        reward = -dt*np.ones(self.state.shape[0])
        wincondition = np.array(self.goal_check()).astype(int)
        reward += wincondition*100
        reward -= not_inside_space*50

        return reward
    
    def goal_check(self):
        position = self.state[:, 0:2]
        wincondition = self.goal.contains(position)
        return wincondition


def compute_concentration(position, center, c0):
    r = np.sqrt((position[:,0]-center[:,0])**2 + (position[:,1]-center[:,1])**2)
    c = c0/(1+r)
    return c
