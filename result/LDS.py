

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
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}  # NEW
            # 立刻落盘一份“当前最佳”
            if save_best:  # NEW
                best_path = os.path.join(save_path, best_ckpt_name)
                torch.save(best_state_dict, best_path)
                print(f"🌟 Updated BEST checkpoint: {best_path} (loss={best_loss:.6f})")
        else:
            no_improve_count += 1
            print(f"⚠️ No improvement for {no_improve_count} epoch(s)")

        if no_improve_count >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
            break

        # 可选：每10个epoch常规保存一次（非最佳）
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

def run_lds_loop(
    train_dataset,
    val_dataset,
    full_model,
    train_loader,
    val_loader,
    tokenizer,
    num_trials=100,
    num_models_per_subset=1,
    subset_ratio=0.5,
    device="cuda",
    do_bootstrap=False,     # CHANGE: 可选——是否算 95% CI
    n_boot=2000,
):
    num_train = len(train_dataset)
    num_test = len(val_dataset)
    print(f"🔁 Running {num_trials} LDS trials...")
    per_z_f = [[] for _ in range(num_test)]  
    per_z_g = [[] for _ in range(num_test)]

    print("Computing TRAK once with full_model ...")
    trak_matrix = get_trak_scores(full_model, train_loader, val_loader)
    trak_matrix = torch.as_tensor(trak_matrix, device='cpu')

    # ✅ 立刻把 full_model 迁到 CPU，释放显存
    full_model.to('cpu')
    del full_model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    for trial in range(num_trials):#对每个样本z取多少次subset
        print(f"\n🎯 Trial {trial + 1}/{num_trials}")

        # 2) 子集选择（保证至少1个）
        subset_size = max(1, int(num_train * subset_ratio))
        subset_indices = random.sample(range(num_train), subset_size)
        subset = Subset(train_dataset, subset_indices)

 
        test_logits = torch.zeros(num_test, device=device)
        for m in range(num_models_per_subset):#对于每个subset来说，训练num_models_per_subset个模型，然后取平均
            model = train_clip_model(subset)
            model.eval()
            for test_idx in range(num_test):
                image, caption = val_dataset[test_idx]
                image = image.to(device)
                text = tokenizer([caption[0]]).to(device)
                sim = encode_similarity(model, image, text.squeeze(0))
                test_logits[test_idx] += sim
        test_logits /= num_models_per_subset  #对于一个样本z来说，f_j(z)是num_models_per_subset个模型输出的平均值
        subset_indices_t = torch.tensor(subset_indices, dtype=torch.long)
        for test_idx in range(num_test):
            tau_z = trak_matrix[:, test_idx]  # [num_train]
            g_j = tau_z.index_select(0, subset_indices_t).sum().item()
            f_j = test_logits[test_idx].item()
            per_z_f[test_idx].append(f_j)#每个per_z_f[test_idx]都是一个list,list长度是num_trials
            per_z_g[test_idx].append(g_j)


    rhos = []
    for test_idx in range(num_test):
        f_series = np.asarray(per_z_f[test_idx], dtype=float)
        g_series = np.asarray(per_z_g[test_idx], dtype=float)
        if len(f_series) >= 2 and len(np.unique(f_series)) > 1 and len(np.unique(g_series)) > 1:
            rho, _ = spearmanr(f_series, g_series)
            rhos.append(rho)

    if len(rhos) == 0:
        print("⚠️ No valid Spearman correlations (try increasing num_trials or varying subsets).")
        return
    print(rhos)
    mean_rho = float(np.mean(rhos))
    print(f"\n✅ LDS (Spearman ρ averaged over {len(rhos)} test examples): {mean_rho:.4f}")
    # if do_bootstrap:
    #     rng = np.random.default_rng(2024)
    #     boots = []
    #     rhos_np = np.array(rhos, dtype=float)
    #     for _ in range(n_boot):
    #         idx = rng.integers(0, len(rhos_np), size=len(rhos_np))
    #         boots.append(float(np.mean(rhos_np[idx])))
    #     lo, hi = np.percentile(boots, [2.5, 97.5])
    #     print(f"95% bootstrap CI over examples: [{lo:.4f}, {hi:.4f}]")

    # # CHANGE: 下面仅作可视化，不用于评分
    # all_f = np.concatenate([np.array(v) for v in per_z_f])
    # all_g = np.concatenate([np.array(v) for v in per_z_g])
    # plt.figure(figsize=(6, 4))
    # plt.scatter(all_f, all_g, alpha=0.5, s=10)
    # plt.xlabel("Model output f_j(z) (CLIP diagonal logits)")
    # plt.ylabel("TRAK prediction g_j(z) (sum over subset)")
    # plt.title(f"LDS scatter (visual only). Mean Spearman ρ = {mean_rho:.3f}")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

train_indices_500 = list(range(5000)) #用来训练full model的训练样本数

train_dataset_500 = Subset(train_dataset_full, train_indices_500)

train_dataset_raw_500 = Subset(train_dataset_raw_full, train_indices_500)

full_model = train_clip_model(train_dataset_500).to(device)
full_model.eval()
N_train=len(train_dataset_raw)
N_val=len(val_dataset_raw)
train_tensor_dataset = build_tensor_dataset(train_dataset_raw, tokenizer, transform_train, N_train)
val_tensor_dataset = build_tensor_dataset(val_dataset_raw, tokenizer, transform_train, N_val)
loader_train_subset_raw = DataLoader(train_tensor_dataset, batch_size=6, shuffle=False,collate_fn=collate_with_index) 
loader_val_subset_raw = DataLoader(val_tensor_dataset, batch_size=6, shuffle=False,collate_fn=collate_with_index)
run_lds_loop(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    full_model=full_model,
    train_loader=loader_train_subset_raw,
    val_loader=loader_val_subset_raw,
    tokenizer=open_clip.get_tokenizer("RN50")
)