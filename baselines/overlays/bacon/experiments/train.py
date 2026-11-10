"""BACON sweep driver: fit one 8-layer multiscale BACON per val mesh, then export
all four output levels with render_sdf.py.

Run from Projects/bacon/experiments after preprocessing/obj_to_xyz.py has written
model_normalized.xyz next to each val mesh. See docs/bacon.md.
"""
import os 
import sys
import subprocess
from pathlib import Path

SHAPENET = Path(__file__).resolve().parents[3] / 'data' / 'shapenet'  # <repo>/data/shapenet


def build_config(hidden_layer, shapenet_file):
    with open(f"./config/sdf/bacon_armadillo.ini", "r") as f:
        config = f.read()
    config = config.replace("hidden_layers = 8", f"hidden_layers = {hidden_layer}")
    config = config.replace("point_cloud_path = ../data/gt_armadillo.xyz", f"point_cloud_path = {shapenet_file}")
    shapenet_name = shapenet_file.split("/")[-3]
    config = config.replace("experiment_name = bacon_armadillo", f"experiment_name = bacon_{shapenet_name}_{hidden_layer}")
    with open(f"./config/sdf/bacon_{shapenet_name}_{hidden_layer}.ini", "w") as f:
        f.write(config)
    return f"./config/sdf/bacon_{shapenet_name}_{hidden_layer}.ini",f"bacon_{shapenet_name}_{hidden_layer}"


def main():
    hidden_layers = [8]

    for hidden_layer in hidden_layers:
        bacon_ckpts = []
        bacon_names = []
        pc_paths = []
        for shapenet_file in open(SHAPENET / 'val.lst'):
            shapenet_file = shapenet_file.strip()
            
            path = f"{SHAPENET}/{shapenet_file}/models/model_normalized.xyz"
            config_path, bacon_name = build_config(hidden_layer, path)

            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_path = os.path.normpath(os.path.join(script_dir, "..", "logs", bacon_name, f"train.log"))
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            proc = subprocess.Popen(
                [sys.executable, "-u", "train_sdf.py", "--config", config_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=script_dir,
                bufsize=1,
            )
            with open(log_path, "w") as log_file:
                for line in proc.stdout:
                    log_file.write(line)
                    log_file.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
            result = proc.wait()
            if result != 0:
                print(f"Failed {bacon_name}, see {log_path}")
                continue
            
            bacon_ckpts.append(f"../logs/{bacon_name}/checkpoints/model_final.pth")
            bacon_names.append(bacon_name)
            pc_paths.append(path)
    
        subprocess.run([
            sys.executable, "render_sdf.py",
            "--bacon-ckpts", ",".join(bacon_ckpts),
            "--bacon-names", ",".join(bacon_names),
            "--point-cloud-paths", ",".join(pc_paths),
        ])

if __name__ == "__main__":
    main()