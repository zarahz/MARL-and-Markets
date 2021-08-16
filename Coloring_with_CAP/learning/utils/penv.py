from multiprocessing import Process, Pipe
import gym
import numpy as np


def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            p = Process(target=worker, args=(remote, env))
            # p.daemon = True
            p.start()
            remote.close()
            self.locals.append(local)

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv()
                                            for local in self.locals]
        return results

    def step(self, joint_actions):
        # send all actions and envs to the worker except the first
        for local, actions in zip(self.locals, (np.array(joint_actions).T)[1:]):
            local.send(("step", actions))
        # execute the first action on the first env manually
        obs, reward, done, info = self.envs[0].step(
            np.array(joint_actions)[:, 0])
        if done:
            obs = self.envs[0].reset()
        # combine parallel envs back into results
        results = zip(*[(obs, reward, done, info)] + [local.recv()
                      for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError
