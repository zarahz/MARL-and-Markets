import csv
import os
import time
from array2gif.core import write_gif
import torch
import logging
import sys

from learning.utils.other import synthesize

# import learning.ppo.utils


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
    create_folders_if_necessary(path)
    return torch.load(path)


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    create_folders_if_necessary(path)
    torch.save(status, path)


def save_capture(model_dir, name, frames):
    path = os.path.join(model_dir, "captures\\"+name)
    create_folders_if_necessary(path)
    print("Saving gif... ", end="")
    write_gif(frames, path, fps=1)
    print("Done.")


def get_model_state(model_dir):
    return get_status(model_dir)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    create_folders_if_necessary(path)

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
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)


def log_stats(logs, key, header, data):
    stats = synthesize(
        logs[key])
    if stats:
        header += [calculation_key + "_" +
                   key for calculation_key in stats.keys()]
        data += stats.values()
    return header, data


def prepare_csv_data(agents, logs, update, num_frames, start_time=None, txt_logger=None):
    # fps = logs["num_frames"]/(update_end_time - update_start_time)
    start_time = start_time if start_time else time.time()
    duration = int(time.time() - start_time)

    header = ["update_count", "frames",
              "duration_in_seconds", "fully_colored"]  # "FPS"
    data = [update, num_frames, duration, logs["fully_colored"]]  # fps

    header, data = log_stats(logs, 'num_reset_fields', header, data)
    header, data = log_stats(logs, 'grid_coloration_percentage', header, data)
    # if "trades" in logs:
    header, data = log_stats(logs, 'trades', header, data)

    if "huber_loss" in logs:
        header, data = log_stats(logs, "huber_loss", header, data)

    all_rewards_per_episode = {}
    for key, value in logs.items():
        if("reward_agent_" in key):
            all_rewards_per_episode[key] = value

    # agent specific data
    for agent in range(agents):
        header, data = log_stats(all_rewards_per_episode,
                                 "reward_agent_" + str(agent), header, data)
        if "entropy" in logs:
            header += ["entropy_agent_" + str(agent)]
            data += [logs["entropy"][agent]]
        if "value" in logs:
            header += ["value_agent_" + str(agent)]
            data += [logs["value"][agent]]
        if "policy_loss" in logs:
            header += ["policy_loss_agent_" + str(agent)]
            data += [logs["policy_loss"][agent]]
        if "grad_norm" in logs:
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
        if "fully_colored" in header or "max" in header or "grad_norm" in header:
            info += "\n"
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
