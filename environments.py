import numpy as np

class BoxEnvironment1():
    def __init__(self, space):
        self.space = space
    
    def init_env(self, agent_batch_size, state_dim, goal):
        self.agent_batch_size = agent_batch_size
        self.state = np.zeros((agent_batch_size,state_dim))
        self.state[:,0] = -0.5*np.ones(agent_batch_size)
        self.goal = goal
    
    def step(self, action, U0, dt, characteristic_length = 1):
        x, y = self.state[:,0], self.state[:,1]
        theta = action[:,0]
        #thermal noise
        noise = np.random.normal(np.zeros(self.agent_batch_size),np.ones(self.agent_batch_size)*np.sqrt(dt))
        # raise Exception(theta.shape, noise.shape)
        theta = theta + np.sqrt(dt)*characteristic_length*noise

        e_x = np.cos(theta)
        v_x = e_x 
        x_new = x + v_x*dt

        e_y = np.sin(theta)
        v_y = e_y 
        y_new = y + v_y*dt
        inside_space = self.space.contains(np.array([x_new, y_new]).T)
        concentration = compute_concentration(np.array([x_new, y_new]).T, self.goal.center, U0)

        self.state[:,0][inside_space] = x_new[inside_space]
        self.state[:,1][inside_space] = y_new[inside_space]
        self.state[:,2] = concentration 
        
        # Compute reward
        reward = self.reward(dt, np.array(inside_space).astype(int))

        return reward
    
    def reward(self, dt, inside_space):
        # Compute reward
        not_inside_space = np.logical_not(inside_space)
        reward = -dt*np.ones(self.state.shape[0])
        wincondition = np.array(self.goal_check()).astype(int)
        reward += wincondition*1000
        reward -= not_inside_space*10
        # reward -= np.linalg.norm(position-self.goal.center, axis=1)*10

        return reward
    
    def goal_check(self):
        position = self.state[:, 0:2]
        wincondition = self.goal.contains(position)
        return wincondition


def compute_concentration(position, center, U0):
    r = np.linalg.norm(position-center, axis=1)
    return U0/(1+r)
