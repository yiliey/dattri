import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CocoCaptions
from tqdm import tqdm



# open_clip 模型和损失函数
import open_clip
from open_clip.loss import ClipLoss

# TRAK attribution
from dattri.algorithm.trak import TRAKAttributor
from dattri.task import AttributionTask
from torch.utils.data import TensorDataset

# 设备设置

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

coco_root = "/mnt/"

train_dataset = CocoCaptions(
    root=os.path.join(coco_root, "train2017"),
    annFile=os.path.join(coco_root, "annotations/captions_train2017.json"),
    transform=transform_train
)
val_dataset = CocoCaptions(
    root=os.path.join(coco_root, "val2017"),
    annFile=os.path.join(coco_root, "annotations/captions_val2017.json"),
    transform=transform_train
)

N = 100 #多了
subset_train_dataset = Subset(train_dataset, indices=list(range(N)))

def collate_fn(batch):
    images, captions = zip(*batch)
    return list(images), list(captions)

train_loader = DataLoader(
    subset_train_dataset,
    batch_size=8, # 用于文章里optimizer的batch size
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn
)

# ====== 损失函数与优化器 ======
loss_fn = ClipLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=5e-5,                  
    weight_decay=0.01,
    betas=(0.9, 0.98)
)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ====== 保存 checkpoint 的目录 ======
os.makedirs("checkpoints", exist_ok=True)

patience = 8       # 连续多少次没有改善就停
min_delta = 1e-3   
best_loss = float('inf')
no_improve_count = 0

# ====== 训练循环 ======
for epoch in range(30):
    model.train()
    total_loss = 0

    for images, captions in train_loader:
        images = torch.stack(images).to(device)  # (B, 3, 224, 224)
        texts = tokenizer([cap[0] for cap in captions]).to(device)

        image_feats, text_feats, logit_scale = model(images, texts)
        loss = loss_fn(image_feats, text_feats, logit_scale)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    lr_scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/100 - Avg Loss: {avg_loss:.4f}")

    # === Early Stopping 判断逻辑 ===
    if best_loss - avg_loss > min_delta:
        best_loss = avg_loss
        no_improve_count = 0
    else:
        no_improve_count += 1
        print(f"⚠️  No improvement for {no_improve_count} epoch(s)")

    if no_improve_count >= patience:
        print(f"⏹️  Early stopping triggered at epoch {epoch+1}")
        break

    # === 保存 checkpoint（每 10 epoch）===
    if (epoch + 1) % 10 == 0:
        ckpt_path = f"checkpoints/clip_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ Saved checkpoint: {ckpt_path}")



train_dataset_raw = CocoCaptions(
    root=os.path.join(coco_root, "train2017"),
    annFile=os.path.join(coco_root, "annotations/captions_train2017.json"),
    transform=None 
)
val_dataset_raw = CocoCaptions(
    root=os.path.join(coco_root, "val2017"),
    annFile=os.path.join(coco_root, "annotations/captions_val2017.json"),
    transform=None
)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4815, 0.4578, 0.4082),
                         std=(0.2686, 0.2613, 0.2758))
])

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

N_train = 10
N_val = 4
def collate_with_index(batch):
    images, captions, indices = zip(*batch)  # unzip
    images = torch.stack(images)
    captions = torch.stack(captions)
    indices = torch.stack(indices)
    # print(f"Batch indices: {indices}")
    return ((images, captions), indices)

train_tensor_dataset = build_tensor_dataset(train_dataset_raw, tokenizer, transform_train, N_train)
val_tensor_dataset = build_tensor_dataset(val_dataset_raw, tokenizer, transform_train, N_val)

loader_train_subset_raw = DataLoader(train_tensor_dataset, batch_size=4, shuffle=False,collate_fn=collate_with_index) 
loader_val_subset_raw = DataLoader(val_tensor_dataset, batch_size=4, shuffle=False,collate_fn=collate_with_index)

data_all=None#可以给一个global变量，代表全数据集
def clip_loss_func(params, data):

    (image, text), index = data # image: [B, 3, 224, 224], text: [B, 77]
    if image.ndim == 3:
        image = image.unsqueeze(0)
    text = text.squeeze(1)
    print(f"[clip_loss_func] image: {image.shape},  text: {text.shape}")

    image_features, text_features, logit_scale = torch.func.functional_call(model, params, (image, text))

    
    print("[🔍 DEBUG] index =", index.detach().cpu())

    # print(f"[clip_loss_func] image_features.shape: {image_features.shape}, norm: {image_features.norm(dim=-1).detach().cpu().tolist()}")
    # print(f"[clip_loss_func] text_features.shape: {text_features.shape}, norm: {text_features.norm(dim=-1).detach().cpu().tolist()}")

    print(f"[clip_loss_func] logit_scale: {logit_scale.item():.4f}")

    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    # 打印 logits 信息
    # print(f"[clip_loss_func] logits_per_image.shape: {logits_per_image.shape}")
    # print(f"[clip_loss_func] logits_per_image: {logits_per_image.detach().cpu()}")
    # print(f"[clip_loss_func] logits_per_image max/min: {logits_per_image.max().item():.4f}/{logits_per_image.min().item():.4f}")

    # print(f"[clip_loss_func] logits_per_text.shape: {logits_per_text.shape}")
    # print(f"[clip_loss_func] logits_per_text: {logits_per_text.detach().cpu()}")

    # 计算 loss
    labels = torch.arange(image.size(0), device=image.device)
    loss_i = nn.CrossEntropyLoss(reduction='none') (logits_per_image, labels) #其实这里应该针对每个sample去有一个loss
    loss_t = nn.CrossEntropyLoss(reduction='none') (logits_per_text, labels)

    total_loss = (loss_i + loss_t)/2 
    if isinstance(index, int):
        index = torch.tensor(index, device=image.device)
    elif isinstance(index, torch.Tensor) and index.ndim == 0:
        index = index.unsqueeze(0)
    
    print(total_loss)

    loss_value = total_loss[index%len(total_loss)]  # index 是 batch 中的第几个 sample
    print(index)
    print(loss_value)
    loss_scalar = loss_value.squeeze()
    print(loss_scalar)
    return loss_scalar



def clip_correct_prob(params, data):
    loss = clip_loss_func(params, data)
    return torch.exp(-loss)
  
task = AttributionTask(
    loss_func=clip_loss_func,
    model=model,
    checkpoints=model.state_dict(),
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
    test_dataloader=loader_val_subset_raw,
    train_dataloader=loader_train_subset_raw
)
print(influences)