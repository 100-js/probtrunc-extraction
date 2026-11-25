import os

data_path = os.path.join("/", "data") # global dataset folder on server

base_path = f"{os.getenv('HOME')}/workspace" # workspace under your username on ubuntu
results_path = os.path.join(base_path, "results", "model_extraction") # where you save your results
ckp_path = os.path.join(results_path, "checkpoints") # checkpoints of trained models
output_path = os.path.join(results_path, "output") 
figure_save_path = os.path.join(output_path, "figures") # where you save your figures
recovery_save_path = os.path.join(output_path, "recovery") # where you save your recovery results