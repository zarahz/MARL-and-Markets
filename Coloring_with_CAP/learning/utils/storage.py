import csv
import os
import time
import torch
import logging
import sys

import learning.utils


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    return "Coloring_with_CAP\\" + get_storage_folder()


def get_storage_folder():
    return "storage"


def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), model_name)


def get_short_model_dir(model_name):
    return os.path.join(get_storage_folder(), model_name)


def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")


def get_status(model_dir):
    # model_dir = 'storage\\one-agent'
    path = get_status_path(model_dir)
    learning.utils.create_folders_if_necessary(path)
    return torch.load(path)


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    learning.utils.create_folders_if_necessary(path)
    torch.save(status, path)


def get_model_state(model_dir):
    return get_status(model_dir)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    learning.utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def get_csv_logger(model_dir, name):
    csv_path = os.path.join(model_dir, name+".csv")
    learning.utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)


def prepare_csv_data(agents, logs, update, num_frames, start_time=None, txt_logger=None):
    # fps = logs["num_frames"]/(update_end_time - update_start_time)
    start_time = start_time if start_time else time.time()
    duration = int(time.time() - start_time)

    header = ["update_count", "frames", "duration_in_seconds"]  # "FPS"
    data = [update, num_frames, duration]  # fps

    reward_per_episode_stats = []  # contains mean,std,min,max
    all_rewards_per_episode = {}
    for key, value in logs.items():
        if("reward_agent_" in key):
            all_rewards_per_episode[key] = value
            reward_per_episode_stats.append(
                learning.utils.synthesize(value))

    # num_frames_per_episode = learning.utils.synthesize(
    #     logs["num_frames_per_episode"])

    # header += ["num_frames_" +
    #            key for key in num_frames_per_episode.keys()]
    # data += num_frames_per_episode.values()

    # agent specific data
    for agent in range(agents):
        header += [key + "_reward_agent_" + str(agent)
                   for key in reward_per_episode_stats[agent].keys()]
        data += reward_per_episode_stats[agent].values()
        header += ["entropy_agent_" + str(agent)]
        data += [logs["entropy"][agent]]
        header += ["value_agent_" + str(agent)]
        data += [logs["value"][agent]]
        header += ["policy_loss_agent_" + str(agent)]
        data += [logs["policy_loss"][agent]]
        header += ["grad_norm_agent_" + str(agent)]
        data += [logs["grad_norm"][agent]]

    if txt_logger:
        print_logs(txt_logger, header, data)

    return header, data, all_rewards_per_episode


def print_logs(txt_logger, header, data):
    info = ""
    for header, value in zip(header, data):
        formatted_value = "{:.2f}".format(
            value) if isinstance(value, float) else str(value)
        info += "| " + header + ": " + formatted_value
    txt_logger.info(info)
    # txt_logger.info(  # FPS {:04.0f} | Frames/Episode : [mean, std, min, Max] {:.1f} {:.1f} {} {}
    #     "Update {} | Frames {:06} | Duration {} | Reward:μσmM {:.2f} {:.2f} {:.2f} {:.2f}".format(*data))
    # txt_logger.info(
    #     str(("Return/Episode/Agent [mean, std, min, Max]: {:.2f} {:.2f} {:.2f} {:.2f}".format(*all_returns_per_episodes))))
    # txt_logger.info(str(("entropy per agent: ", logs["entropy"])))
    # txt_logger.info(str(("value per agent: ", logs["value"])))
    # txt_logger.info(
    #     str(("value loss per agent: ", logs["value_loss"])))
    # txt_logger.info(str(("grad norm per agent: ", logs["grad_norm"])))

    # txt_logger.info(
    #     "Update {} | Frames {:06} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
    #     .format(*data))


def update_csv_file(file, logger, update, content):
    if update == 1:  # status["num_frames"] == 0:
        logger.writerow(list(content.keys()))
    logger.writerow(list(content.values()))
    file.flush()
