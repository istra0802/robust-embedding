"""
Step 2 - train a clean TransE model on data/UMLS_clean using dicee.

Usage:
    python scripts/03_train_clean.py

Produces an Experiments/<timestamp> folder (dicee's default output layout)
which we point to explicitly via --path_to_store_single_run so downstream
scripts can find it deterministically.
"""
from dicee.executer import Execute
from dicee.config import Namespace

args = Namespace()
args.model = "TransE"
args.dataset_dir = "data/UMLS_clean"
args.path_to_store_single_run = "Experiments/UMLS_TransE_clean"
args.scoring_technique = "KvsAll"          # matches paper-style 1-K training protocol
args.embedding_dim = 32
args.num_epochs = 256
args.batch_size = 1024
args.lr = 0.01
args.eval_model = "train_val_test"
args.random_seed = 1

if __name__ == "__main__":
    reports = Execute(args).start()
    print("Train MRR:", reports["Train"]["MRR"])
    print("Val   MRR:", reports["Val"]["MRR"])
    print("Test  MRR:", reports["Test"]["MRR"])
