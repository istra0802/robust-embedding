"""
Step 7 - retrain TransE on a poisoned dataset with EXACTLY the same
hyperparameters as the clean run (paper Section 4: "All hyperparameters
for training on original and poisoned data remain the same").

Usage:
    python scripts/08_train_poisoned.py data/UMLS_poisoned_symmetry_truth \
        Experiments/UMLS_TransE_poisoned_symmetry_truth
"""
import sys
from dicee.executer import Execute
from dicee.config import Namespace

dataset_dir = sys.argv[1]
out_dir = sys.argv[2]

args = Namespace()
args.model = "TransE"
args.dataset_dir = dataset_dir
args.path_to_store_single_run = out_dir
args.scoring_technique = "KvsAll"
args.embedding_dim = 32
args.num_epochs = 256
args.batch_size = 1024
args.lr = 0.01
args.eval_model = "train_val_test"
args.random_seed = 1   # same seed as clean run -> isolates the effect of poisoning

if __name__ == "__main__":
    reports = Execute(args).start()
    print("Train MRR:", reports["Train"]["MRR"])
    print("Val   MRR:", reports["Val"]["MRR"])
    print("Test  MRR:", reports["Test"]["MRR"])
