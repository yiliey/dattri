
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LinearRegression
import open_clip
from typing import Union
from torch.nn import Module
import torch.optim as optim

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CocoCaptions
from tqdm import tqdm
from scipy.stats import spearmanr



# open_clip 模型和损失函数
import open_clip
from open_clip.loss import ClipLoss

# TRAK attribution
from dattri.algorithm.trak import TRAKAttributor
from dattri.task import AttributionTask
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
# ===== 1. Load CLIP model and tokenizer =====
model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
model = model.to(device)
tokenizer = open_clip.get_tokenizer("RN50")

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4815, 0.4578, 0.4082),
                         std=(0.2686, 0.2613, 0.2758))
])

coco_root = "mscoco2017/"

train_dataset_full = CocoCaptions(
    root=os.path.join(coco_root, "train2017"),
    annFile=os.path.join(coco_root, "annotations/captions_train2017.json"),
    transform=transform_train
)
val_dataset_full = CocoCaptions(
    root=os.path.join(coco_root, "val2017"),
    annFile=os.path.join(coco_root, "annotations/captions_val2017.json"),
    transform=transform_train
)
train_dataset_raw_full = CocoCaptions(
    root=os.path.join(coco_root, "train2017"),
    annFile=os.path.join(coco_root, "annotations/captions_train2017.json"),
    transform=None 
)
val_dataset_raw_full = CocoCaptions(
    root=os.path.join(coco_root, "val2017"),
    annFile=os.path.join(coco_root, "annotations/captions_val2017.json"),
    transform=None
)
random.seed(42)

train_indices = random.sample(range(len(train_dataset_full)), 5000)
val_indices = random.sample(range(len(val_dataset_full)), 500)

train_dataset = Subset(train_dataset_full, train_indices)
val_dataset = Subset(val_dataset_full, val_indices)

train_dataset_raw = Subset(train_dataset_raw_full, train_indices)
val_dataset_raw = Subset(val_dataset_raw_full, val_indices)

# ========== 工具函数 ==========
def collate_fn(batch):
    images, captions = zip(*batch)
    return list(images), list(captions)

def collate_with_index(batch):
    images, captions, indices = zip(*batch)  # unzip
    images = torch.stack(images)
    captions = torch.stack(captions)
    indices = torch.stack(indices)
    # print(f"Batch indices: {indices}")
    return ((images, captions), indices)


def build_tensor_dataset(coco_dataset, tokenizer, transform, N):
    image_tensor_list = []
    caption_tensor_list = []
    index_tensor_list = []

    for i in tqdm(range(N), desc="Preprocessing dataset"):
        img, captions = coco_dataset[i]


        img_tensor = transform(img)  # shape: (3, 224, 224)
        image_tensor_list.append(img_tensor)
        text_tensor = tokenizer(captions[0])  # shape: (1, 77)
        caption_tensor_list.append(text_tensor)
        index_tensor_list.append(torch.tensor(i))

    image_tensor_stack = torch.stack(image_tensor_list)  # shape: [N, 3, 224, 224]
    caption_tensor_stack = torch.stack(caption_tensor_list)  # shape: [N, 1, 77]
    index_tensor_stack = torch.stack(index_tensor_list)     

    # ==== 🔍 最终输出检查 ====
    print(f"\n✅ Done stacking {N} samples:")
    print(f"  📦 image_tensor_stack shape: {image_tensor_stack.shape}, dtype: {image_tensor_stack.dtype}")
    print(f"  📦 caption_tensor_stack shape: {caption_tensor_stack.shape}, dtype: {caption_tensor_stack.dtype}")

    return TensorDataset(image_tensor_stack, caption_tensor_stack, index_tensor_stack)

def clip_loss_func(params, data):
    model.train(False) 
    (image, text), index = data # image: [B, 3, 224, 224], text: [B, 77]
    if image.ndim == 3:
        image = image.unsqueeze(0)
    text = text.squeeze(1)
    print(f"[clip_loss_func] image: {image.shape},  text: {text.shape}")
    

    image_features, text_features, logit_scale = torch.func.functional_call(model, params, (image, text))


    # print(f"[clip_loss_func] image_features.shape: {image_features.shape}, norm: {image_features.norm(dim=-1).detach().cpu().tolist()}")
    # print(f"[clip_loss_func] text_features.shape: {text_features.shape}, norm: {text_features.norm(dim=-1).detach().cpu().tolist()}")

    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    # 计算 loss
    labels = torch.arange(image.size(0), device=image.device)
    loss_i = nn.CrossEntropyLoss(reduction='none') (logits_per_image, labels) #其实这里应该针对每个sample去有一个loss
    loss_t = nn.CrossEntropyLoss(reduction='none') (logits_per_text, labels)

    total_loss = (loss_i + loss_t)/2 
    if isinstance(index, int):
        index = torch.tensor(index, device=image.device)
    elif isinstance(index, torch.Tensor) and index.ndim == 0:
        index = index.unsqueeze(0)
    loss_value = total_loss[index%len(total_loss)]  # index 是 batch 中的第几个 sample
    loss_scalar = loss_value.squeeze()
    return loss_scalar



def clip_correct_prob(params, data):
    loss = clip_loss_func(params, data)
    return torch.exp(-loss)


# ========== 模型训练函数==========
def train_clip_model(
    dataset: Union[Subset, torch.utils.data.Dataset],
    num_epochs: int = 30,
    batch_size: int = 8,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    patience: int = 8,
    min_delta: float = 1e-3,
    save_checkpoints: bool = False,
    save_path: str = "checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_best: bool = True,                    
    best_ckpt_name: str = "clip_best.pt"
) -> Module:
    model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer("RN50")
    loss_fn = ClipLoss()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98)
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    if save_checkpoints or save_best:           
        os.makedirs(save_path, exist_ok=True)   

    best_loss = float("inf")
    no_improve_count = 0
    best_state_dict = None 

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, captions in loader:
            images = torch.stack(images).to(device)
            texts = tokenizer([random.choice(cap) for cap in captions]).to(device)

            image_feats, text_feats, logit_scale = model(images, texts)
            loss = loss_fn(image_feats, text_feats, logit_scale)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        lr_scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            no_improve_count = 0
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} 
            if save_best: 
                best_path = os.path.join(save_path, best_ckpt_name)
                torch.save(best_state_dict, best_path)
                print(f"🌟 Updated BEST checkpoint: {best_path} (loss={best_loss:.6f})")
        else:
            no_improve_count += 1
            print(f"⚠️ No improvement for {no_improve_count} epoch(s)")

        if no_improve_count >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
            break

        if save_checkpoints and (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(save_path, f"clip_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Saved checkpoint: {ckpt_path}")

    if best_state_dict is not None:            
        model.load_state_dict(best_state_dict)  
        model = model.to(device)               
        print(f"🎯 Loaded BEST model (loss={best_loss:.6f}) for return.")  
    else:
        print("ℹ️ No improvement recorded; returning last-epoch model.")  
    return model

# ========== 图文相似度计算 ==========
def encode_similarity(model, image, text):
    with torch.no_grad():
        img_f = model.encode_image(image.unsqueeze(0))
        txt_f = model.encode_text(text.unsqueeze(0))
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        sim = (logit_scale * (img_f @ txt_f.T)).diag().item()
    return sim

def get_trak_scores(model, train_loader,val_loader):
    ckpt_on_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    task = AttributionTask(
    loss_func=clip_loss_func,
    model=model,
    checkpoints=ckpt_on_cpu,
    batch_mode=True # 设置为 True 以启用批处理模式 
    )
    trak = TRAKAttributor(
        task=task,
        correct_probability_func=clip_correct_prob,
        projector_kwargs={"proj_dim": 128},
        layer_name=None,
        device=device,
        regularization=1e-5,
        batch_mode=True # 设置为 True 以启用批处理模式
    ) 


    influences = trak.attribute(
        test_dataloader=val_loader,
        train_dataloader=train_loader
    )
    print(influences)
    return influences


def show_test_and_topk_train(
    influences, target_idx, 
    train_subset, train_dataset_raw_full,
    val_subset, val_dataset_raw_full,
    k=5,
    save_path: str = None,   
    show: bool = True      
):
    val_raw_idx = val_subset.indices[target_idx]
    test_img, test_caps = val_dataset_raw_full[val_raw_idx]

    col = influences[:, target_idx]

    vals_top, idxs_top = torch.topk(col, k=k)
    vals_bot, idxs_bot = torch.topk(-col, k=k) 

    plt.figure(figsize=(4*(k+1), 8))

    ax = plt.subplot(2, k+1, 1)
    ax.imshow(test_img)
    ax.set_title(f"Test #{target_idx}\n{test_caps[0]}", fontsize=9)
    ax.axis("off")

    for j, (i, v) in enumerate(zip(idxs_top.tolist(), vals_top.tolist()), 2):
        raw_idx = train_subset.indices[i]
        img, caps = train_dataset_raw_full[raw_idx]
        ax = plt.subplot(2, k+1, j)
        ax.imshow(img)
        ax.set_title(f"Top {j-1}: Train#{i}\n{v:.4f}", fontsize=9)
        ax.axis("off")

    ax = plt.subplot(2, k+1, k+2)
    ax.imshow(test_img)
    ax.set_title(f"Test #{target_idx}\n{test_caps[0]}", fontsize=9)
    ax.axis("off")

    for j, (i, v) in enumerate(zip(idxs_bot.tolist(), (-vals_bot).tolist()), 2):
        raw_idx = train_subset.indices[i]
        img, caps = train_dataset_raw_full[raw_idx]
        ax = plt.subplot(2, k+1, k+1+j)
        ax.imshow(img)
        ax.set_title(f"Bot {j-1}: Train#{i}\n{v:.4f}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    # === 保存文件 ===
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def save_topk_for_all_tests(
    influences: torch.Tensor,
    train_subset,
    train_dataset_raw_full,
    val_subset,
    val_dataset_raw_full,
    k: int = 5,
    out_dir: str = "results",
    filename_tpl: str = "top{K}_{idx}.png",   # 命名模板：会用 K 和 idx 替换
    show: bool = False,
    indices: list = None                      # 可选：只导出这些 test indices；默认导出所有
):
    os.makedirs(out_dir, exist_ok=True)

    n_train, n_val = influences.shape
    if len(train_subset) != n_train:
        raise ValueError(f"train_subset size {len(train_subset)} != influences rows {n_train}")
    if len(val_subset) != n_val:
        raise ValueError(f"val_subset size {len(val_subset)} != influences cols {n_val}")

    if indices is None:
        indices = list(range(n_val))

    for idx in tqdm(indices, desc=f"Saving top-{k} figures for tests"):
        save_path = os.path.join(out_dir, filename_tpl.format(K=k, idx=idx))
        try:
            show_test_and_topk_train(
                influences=influences,
                target_idx=idx,
                train_subset=train_subset,
                train_dataset_raw_full=train_dataset_raw_full,
                val_subset=val_subset,
                val_dataset_raw_full=val_dataset_raw_full,
                k=k,
                save_path=save_path,
                show=show
            )
        except Exception as e:
            print(f"[warn] failed on test idx={idx}: {e}")

train_indices_5000 = list(range(5000)) #用来训练full model的训练样本数

train_dataset_5000 = Subset(train_dataset_full, train_indices_5000)

train_dataset_raw_5000 = Subset(train_dataset_raw_full, train_indices_5000)

full_model = train_clip_model(train_dataset_5000).to(device)
full_model.eval()

N_train=len(train_dataset_raw)
N_val=len(val_dataset_raw)

train_tensor_dataset = build_tensor_dataset(train_dataset_raw, tokenizer, transform_train, N_train)
val_tensor_dataset = build_tensor_dataset(val_dataset_raw, tokenizer, transform_train, N_val)

loader_train_subset_raw = DataLoader(train_tensor_dataset, batch_size=6, shuffle=False,collate_fn=collate_with_index) 
loader_val_subset_raw = DataLoader(val_tensor_dataset, batch_size=6, shuffle=False,collate_fn=collate_with_index)

influences = get_trak_scores(full_model, loader_train_subset_raw, loader_val_subset_raw)
influences= torch.as_tensor(influences)

full_model.to('cpu')
del full_model
torch.cuda.empty_cache()
import gc; gc.collect()


save_topk_for_all_tests(
    influences=influences,
    train_subset=train_dataset,
    train_dataset_raw_full=train_dataset_raw_full,
    val_subset=val_dataset,
    val_dataset_raw_full=val_dataset_raw_full,
    k=5,
    out_dir="results",
    filename_tpl="top{K}_{idx}.png",
    show=False
)
