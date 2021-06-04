from model_wrapper import ModelForSCMomentum, ModelForSCDualMomentum
from dataset import LRADataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import os
import json
import pickle
import numpy as np
import argparse
import math
import itertools
import lra_config

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
parser.add_argument("--task", type = str, help = "task", dest = "task", required = True)
parser.add_argument("--skip_train", type = int, help = "skip_train", dest = "skip_train", default = 0)
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--diag_size', type=int, default=1,
                    help='diag size for sparse transformer')
parser.add_argument(
            "--sparse_ratio",
            type=float,
            default=0.5,
            help="ratio between sparse and lowrank"
        )
parser.add_argument('--kernels', type=str, nargs='+', default=['elu',],
                        help='kernels to use for lowrank.')
parser.add_argument('--project_name', type=str, default=None,
                    help='project name for wandb.')
parser.add_argument('--job_name', type=str, default=None,
                    help='job name for wandb.')
parser.add_argument('--use_wandb', action='store_true',
                    help='use wandb.')
parser.add_argument(
        "--mu",
        type=float,
        default=0.0,
        help="Momentum to be used for momentum transformer layers"
    )
parser.add_argument(
        "--stepsize",
        type=float,
        default=1.0,
        help="Stepsize to be used for momentum transformer layers"
    )
parser.add_argument(
        "--res_stepsize",
        type=float,
        default=1.0,
        help="Stepsize to be used for momentum residual connections"
    )
parser.add_argument(
        "--res_delta",
        type=float,
        default=0.0001,
        help="Delta to be used for momentum residual connections"
    )
parser.add_argument(
        "--res_mu",
        type=float,
        default=0.0,
        help="Momentum to be used for momentum residual connections"
    )

args = parser.parse_args()

attn_type = args.model
task = args.task

checkpoint_dir = "/tanData/momentum_transformer/lra/"

if args.use_wandb:  # configure wandb.
    import wandb
    use_wandb = True

    if args.project_name is None:
        project_name = (os.uname()[1]
                        + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        project_name = args.project_name

    wandb.init(project=project_name)

    if args.job_name is None:
        # wandb.run.name = (os.uname()[1]
        #                   + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #                   + args.work_dir)
        wandb.run.name = f"{os.uname()[1]}//{args.attn_type}//" \
                         f"{args.performer_proj_dim}//" \
                         f"{args.tgt_len}-{args.eval_tgt_len}" \
                         f"-{args.mem_len}//" \
                         f"{args.n_layer}-{args.n_head}-" \
                         f"{args.d_head}-{args.d_embed}-{args.d_model}-" \
                         f"{args.d_inner}-{args.dropout}-{args.dropatt}-//" \
                         f"{args.lr}-{args.warmup_step}" \
                         f"{args.batch_size}-{args.eval_batch_size}//" \
                         f"{args.seed}-{args.work_dir}-{args.dpfp_n_roll}" \
                         f"-{args.carry_over_fast_weight}-{args.no_pos}"
    else:
        wandb.run.name = f"{os.uname()[1]}//{args.job_name}"

    wandb_config = wandb.config
    wandb_config.host = os.uname()[1]  # host node name
    wandb_config.diag_size = args.diag_size
    wandb_config.sparse_ratio = args.sparse_ratio
    wandb_config.kernels = args.kernels
    wandb_config.mu = args.mu
    wandb_config.stepsize = args.stepsize
    wandb_config.res_stepsize = args.res_stepsize
    wandb_config.res_delta = args.res_delta
    wandb_config.res_mu = args.res_mu
else:
    use_wandb = False

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

print(lra_config.config[task]["extra_attn_config"].keys(), flush = True)

model_config = lra_config.config[task]["model"]
model_config.update(lra_config.config[task]["extra_attn_config"][attn_type])

model_config["mixed_precision"] = True
model_config["attn_type"] = attn_type
model_config["max_seq_len"] = int(2 ** math.ceil(math.log2(model_config["max_seq_len"])))
model_config["diag_size"] = args.diag_size
model_config["sparse_ratio"] = args.sparse_ratio
model_config["kernels"] = args.kernels
model_config["mu"] = args.mu
model_config["stepsize"] = args.stepsize
model_config["res_stepsize"] = args.res_stepsize
model_config["res_delta"] = args.res_delta
model_config["res_mu"] = args.res_mu

training_config = lra_config.config[task]["training"]
gpu_memory_config = lra_config.config[task]["gpu_memory"]

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, training_config], indent = 4))

if task == "retrieval":
    model = ModelForSCDualMomentum(model_config)
else:
    model = ModelForSCMomentum(model_config)

print(model)
print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

ds_iter = {
    "train":enumerate(DataLoader(LRADataset(f"../datasets/{task}.train.pickle", True), batch_size = training_config["batch_size"], drop_last = True)),
    "dev":enumerate(DataLoader(LRADataset(f"../datasets/{task}.dev.pickle", True), batch_size = training_config["batch_size"], drop_last = True)),
    "test":enumerate(DataLoader(LRADataset(f"../datasets/{task}.test.pickle", False), batch_size = training_config["batch_size"], drop_last = True)),
}

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = training_config["learning_rate"],
    betas = (0.9, 0.999), eps = 1e-6, weight_decay = training_config["weight_decay"]
)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer = optimizer,
    max_lr = training_config["learning_rate"],
    pct_start = training_config["warmup"] / training_config["num_train_steps"],
    anneal_strategy = training_config["lr_decay"],
    total_steps = training_config["num_train_steps"]
)

amp_scaler = torch.cuda.amp.GradScaler() if model_config["mixed_precision"] else None

def step(component, step_idx):
    t0 = time.time()

    optimizer.zero_grad()

    _, batch = next(ds_iter[component])
    for key in batch:
        batch[key] = batch[key].cuda()

    if component == "train":
        outputs = {}

        partial_inputs_list = [{} for _ in range(accumu_steps)]
        for key in batch:
            for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                partial_inputs_list[idx][key] = inp

        for partial_inputs in partial_inputs_list:
            partial_outputs = model(**partial_inputs)
            for key in partial_outputs:
                partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                if key not in outputs:
                    outputs[key] = partial_outputs[key]
                else:
                    outputs[key] += partial_outputs[key]
            amp_scaler.scale(partial_outputs["loss"]).backward()

        amp_scaler.step(optimizer)
        amp_scaler.update()
        lr_scheduler.step()
    else:
        with torch.no_grad():
            outputs = {}

            partial_inputs_list = [{} for _ in range(accumu_steps)]
            for key in batch:
                for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                    partial_inputs_list[idx][key] = inp

            for partial_inputs in partial_inputs_list:
                partial_outputs = model(**partial_inputs)
                for key in partial_outputs:
                    partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                    if key not in outputs:
                        outputs[key] = partial_outputs[key]
                    else:
                        outputs[key] += partial_outputs[key]

    t1 = time.time()

    batch_size = batch[list(batch.keys())[0]].size(0)
    t_escape = t1 - t0
    learning_rate = optimizer.param_groups[0]["lr"]
    loss = outputs["loss"].data.item()
    accu = outputs["accu"].data.item()
    time_since_start = time.time() - init_t

    print(f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.4f}, accu={accu:.4f}\t\t\t\t", end = "\r", flush = True)

    summary[component]["t"] += t_escape
    summary[component]["loss"].append(loss)
    summary[component]["accu"].append(accu)

def print_summary(summary, save_if_improved, train_step_idx):
    summary["loss"] = np.mean(summary["loss"])
    summary["accu"] = np.mean(summary["accu"])

    print()
    if summary["accu"] > summary["best_accu"]:
        summary["best_accu"] = summary["accu"]
        if save_if_improved:
            best_accu = summary["best_accu"]
            torch.save({"model_state_dict":model.module.state_dict()}, log_f_path.replace(".log", ".model"))
            print(f"best_accu={best_accu}. Saved best model")

    summary_round = {"train_step_idx":train_step_idx}
    for key in summary:
        if type(summary[key]) is str:
            summary_round[key] = summary[key]
        else:
            summary_round[key] = round(summary[key], 4)

    print(summary_round, flush = True)
    log_f.write(json.dumps(summary_round, sort_keys = True) + "\n")
    log_f.flush()

    summary["t"] = 0
    summary["loss"] = []
    summary["accu"] = []

init_t = time.time()

log_f_path = os.path.join(checkpoint_dir, f"{args.job_name}_output.log")
log_f = open(log_f_path, "a+")

summary = {
    component:{"t":0, "loss":[], "accu":[], "best_accu":0, "component":component}
    for component in ["train", "dev", "test"]
}

accumu_steps = max(training_config["batch_size"] // len(device_ids) // gpu_memory_config[attn_type], 1)
print(f"accumu_steps={accumu_steps}")

if args.skip_train == 0:
    try:
        model.train()
        for train_step_idx in range(training_config["num_train_steps"]):
            outputs = step("train", train_step_idx)

            if (train_step_idx + 1) % training_config["eval_frequency"] == 0:
                if use_wandb:
                    wandb.log({"loss_train": np.mean(summary["train"]["loss"])})
                    wandb.log({"accu_train": np.mean(summary["train"]["accu"])})
                print_summary(summary["train"], False, train_step_idx)
                model.eval()
                for dev_step_idx in range(training_config["num_eval_steps"]):
                    outputs = step("dev", dev_step_idx)
                if use_wandb:
                    wandb.log({"loss_dev": np.mean(summary["dev"]["loss"])})
                    wandb.log({"accu_dev": np.mean(summary["dev"]["accu"])})
                print_summary(summary["dev"], True, train_step_idx)
                model.train()
    except KeyboardInterrupt as e:
        print(e)

checkpoint = torch.load(log_f_path.replace(".log", ".model"), map_location = "cpu")
model.module.load_state_dict(checkpoint["model_state_dict"])
model.eval()
try:
    for test_step_idx in itertools.count():
        outputs = step("test", test_step_idx)
    print_summary(summary["test"], False, train_step_idx)
    if use_wandb:
        wandb.log({"loss_test": np.mean(summary["test"]["loss"])})
        wandb.log({"accu_test": np.mean(summary["test"]["accu"])})
except StopIteration:
    if use_wandb:
        wandb.log({"loss_test": np.mean(summary["test"]["loss"])})
        wandb.log({"accu_test": np.mean(summary["test"]["accu"])})
    print_summary(summary["test"], False, train_step_idx)
