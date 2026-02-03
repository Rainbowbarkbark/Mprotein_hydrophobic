import torch
from transformers import AutoModelForMaskedLM

# MODEL_ID = "Synthyra/ESMplusplus_large"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL_PRETRAINED = (AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True)).to(DEVICE).eval()
# TOKENIZER = MODEL_PRETRAINED.tokenizer


def embedding_extract(model, tokenizer, sequence, device):
    X_tokenized = tokenizer(
        sequence,
        padding=True,
        return_tensors="pt",
        # max_length = max_length
    )
    attention_mask = X_tokenized["attention_mask"]
    X_tokenized = X_tokenized.to(device)
    with torch.no_grad():
        output = model(**X_tokenized, output_hidden_states=True)
    X_embeddings = output.hidden_states[-1].squeeze(0).cpu()
    X_embeddings = X_embeddings[:, 1:-1, :]
    attention_mask = attention_mask.cpu()
    return X_embeddings, attention_mask


def attention_map_extract(model, tokenizer, sequence, device):
    X_tokenized = tokenizer(
        sequence,
        padding=True,
        return_tensors="pt",
    )
    X_tokenized = X_tokenized.to(device)

    with torch.no_grad():
        output = model(**X_tokenized, output_attentions=True)
        M = torch.stack([i.squeeze(0).cpu() for i in output.attentions])
        M_sym = (M + M.transpose(-2, -1)) / 2
        F_i = M_sym.sum(dim=-1, keepdim=True)
        F_j = M_sym.sum(dim=-2, keepdim=True)
        F_total = M_sym.sum(dim=(-1, -2), keepdim=True)
        F_APC = M_sym - (F_i @ F_j) / (F_total)
    return F_APC.contiguous().clone()
