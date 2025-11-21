import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_RBC
from dataset_RBC import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI RBC")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for training/eval')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--downsample", type=int, default=64, help="spatial downsample size (square)")
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--use_physics", action="store_true", help="Enable dataset physics loss")
parser.add_argument("--lambda_phys", type=float, default=1.0, help="Weight for physics loss term")

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/rbc_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# downsample size -> (h,w)
downsample_size = (args.downsample, args.downsample)

train_loader, valid_loader, test_loader, mean, std = get_dataloader(
    batch_size=config["train"]["batch_size"],
    seed=args.seed,
    downsample_size=downsample_size,
)

# target_dim is number of features per time step (length of mean)
target_dim = len(mean)

model = CSDI_RBC(
    config,
    args.device,
    target_dim=target_dim,
    use_physics=args.use_physics,
    lambda_phys=args.lambda_phys,
    mean=mean,
    std=std,
).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
