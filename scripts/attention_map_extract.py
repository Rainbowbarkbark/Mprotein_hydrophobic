import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import h5py
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForMaskedLM
from scripts.pretrained_esmc import attention_map_extract
from scripts.pdb_utils import extract_contact_map
from src.utils import save_dataset

MODEL_ID = "Synthyra/ESMplusplus_large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PRETRAINED = (
    (AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True))
    .to(DEVICE)
    .eval()
)
TOKENIZER = MODEL_PRETRAINED.tokenizer

df_raw = pd.read_csv("./data/raw/ACC_PDB_preprocess_2.csv")
df = df_raw.copy()

with h5py.File("./data/maps/maps.h5", "a") as f:
    for i in tqdm(range(100)):
        acc = df["ACC"][i]
        pdb_id = df["PDB_ID"][i]
        cmap, seq = extract_contact_map(pdb_id, df["Target_Chain"][i], threshold=8.0)
        if cmap is None:
            continue
        amap = attention_map_extract(MODEL_PRETRAINED, TOKENIZER, seq, DEVICE)
        grp = f.require_group(acc)
        save_dataset(grp, "attention_map", amap, opts=4)
        save_dataset(grp, "contact_map", cmap, opts=9)
