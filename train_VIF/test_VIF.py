"""
Complete VIF-CG Implementation for CLIP
Integrates with your existing CLIP training code
"""

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch import nn
from torchvision import transforms
from torchvision.datasets import CocoCaptions
from tqdm import tqdm
import open_clip


# ==================== VIF-CG Attributor ====================

class VIF_CG_Attributor:
    """
    Versatile Influence Function (VIF) with Conjugate Gradient for CLIP.
    """
    
    def __init__(self, model, device='cuda', proj_dim=512, regularization=1e-5, 
                 cg_max_iter=100, cg_tol=1e-5):
        self.model = model
        self.device = device
        self.proj_dim = proj_dim
        self.regularization = regularization
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.model.eval()
        
        self.num_params = sum(p.numel() for p in model.parameters())
        print(f"[VIF-CG] Model parameters: {self.num_params:,}")
        print(f"[VIF-CG] Projection dimension: {proj_dim}")
        
        # Random projection
        self.projection_matrix = torch.randn(
            self.num_params, self.proj_dim, device=self.device
        ) / np.sqrt(self.proj_dim)
    
    def _flatten_grads(self, grad_list):
        return torch.cat([g.flatten() for g in grad_list])
    
    def _project(self, vector):
        return vector @ self.projection_matrix
    
    def compute_clip_loss(self, images, texts, mask=None):
        """Compute CLIP contrastive loss."""
        image_features, text_features, logit_scale = self.model(images, texts)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Apply mask
        if mask is not None:
            valid_idx = (mask == 1).nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0:
                return torch.tensor(0.0, device=self.device)
            image_features = image_features[valid_idx]
            text_features = text_features[valid_idx]
        
        # Compute logits
        logits_per_image = logit_scale * (image_features @ text_features.T)
        logits_per_text = logits_per_image.T
        
        # Cross-entropy loss
        B = image_features.size(0)
        labels = torch.arange(B, device=self.device)
        loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2
    
    def _compute_gradient(self, train_loader, exclude_idx=None):
        """Compute gradient with optional sample exclusion."""
        self.model.zero_grad()
        total_loss = 0.0
        total_count = 0
        
        for batch_data, batch_indices in train_loader:
            images, texts = batch_data
            images = images.to(self.device)
            texts = texts.squeeze(1).to(self.device)
            batch_indices = batch_indices.to(self.device)
            
            # Create mask
            if exclude_idx is not None:
                mask = (batch_indices != exclude_idx).float()
            else:
                mask = torch.ones(images.size(0), device=self.device)
            
            if mask.sum() == 0:
                continue
            
            # Compute loss
            loss = self.compute_clip_loss(images, texts, mask=mask)
            total_loss += loss * mask.sum()
            total_count += mask.sum()
        
        # Backward
        if total_count > 0:
            avg_loss = total_loss / total_count
            avg_loss.backward()
        
        # Extract gradients
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.clone().detach())
            else:
                grads.append(torch.zeros_like(param))
        
        return grads
    
    def compute_gradient_difference(self, train_loader, exclude_idx):
        """Compute ∇[L(θ, 1) - L(θ, 1^(-i))]"""
        grad_full = self._compute_gradient(train_loader, exclude_idx=None)
        grad_exclude = self._compute_gradient(train_loader, exclude_idx=exclude_idx)
        
        grad_diff_list = [g_full - g_excl for g_full, g_excl in zip(grad_full, grad_exclude)]
        grad_diff = self._flatten_grads(grad_diff_list)
        
        return grad_diff
    
    def hessian_vector_product(self, train_loader, vector):
        """Compute H·v using double backward."""
        hvp = torch.zeros_like(vector)
        total_count = 0
        
        for batch_data, _ in train_loader:
            images, texts = batch_data
            images = images.to(self.device)
            texts = texts.squeeze(1).to(self.device)
            batch_size = images.size(0)
            
            self.model.zero_grad()
            loss = self.compute_clip_loss(images, texts)
            
            # First backward
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            flat_grads = self._flatten_grads(list(grads))
            
            # grad^T · v
            grad_v = torch.dot(flat_grads, vector)
            
            # Second backward
            hvp_grads = torch.autograd.grad(grad_v, self.model.parameters())
            batch_hvp = self._flatten_grads(list(hvp_grads))
            
            hvp += batch_hvp * batch_size
            total_count += batch_size
        
        hvp = hvp / total_count + self.regularization * vector
        return hvp
    
    def conjugate_gradient(self, train_loader, b):
        """Solve H·x = b using Conjugate Gradient."""
        print(f"[CG] Solving H·x = b (dim={len(b):,})...")
        
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rs_old = torch.dot(r, r)
        
        for i in range(self.cg_max_iter):
            Hp = self.hessian_vector_product(train_loader, p)
            
            pHp = torch.dot(p, Hp)
            alpha = rs_old / (pHp + 1e-10)
            
            x = x + alpha * p
            r = r - alpha * Hp
            rs_new = torch.dot(r, r)
            
            residual_norm = torch.sqrt(rs_new).item()
            if i % 10 == 0 or residual_norm < self.cg_tol:
                print(f"  Iter {i+1}: residual = {residual_norm:.2e}")
            
            if residual_norm < self.cg_tol:
                print(f"[CG] Converged at iteration {i+1}")
                break
            
            beta = rs_new / (rs_old + 1e-10)
            p = r + beta * p
            rs_old = rs_new
        
        return x
    
    def attribute(self, train_loader, test_loader):
        """Compute VIF influence scores."""
        n_train = len(train_loader.dataset)
        n_test = len(test_loader.dataset)
        
        print(f"\n{'='*70}")
        print(f"VIF-CG Attribution: {n_train} train × {n_test} test")
        print(f"{'='*70}\n")
        
        influences = np.zeros((n_train, n_test))
        
        # Step 1: Training gradient differences
        print("Step 1/3: Computing training gradient differences...")
        train_grads = []
        
        for i in tqdm(range(n_train), desc="Train gradients"):
            grad_diff = self.compute_gradient_difference(train_loader, exclude_idx=i)
            proj_grad = self._project(grad_diff)
            train_grads.append(proj_grad.cpu())
        
        train_grads = torch.stack(train_grads).to(self.device)
        
        # Step 2: Test gradients
        print("\nStep 2/3: Computing test gradients...")
        test_grads = []
        
        for batch_data, _ in tqdm(test_loader, desc="Test gradients"):
            images, texts = batch_data
            images = images.to(self.device)
            texts = texts.squeeze(1).to(self.device)
            
            self.model.zero_grad()
            loss = self.compute_clip_loss(images, texts)
            loss.backward()
            
            grads = [p.grad.clone().detach() for p in self.model.parameters() 
                    if p.grad is not None]
            grad_vector = self._flatten_grads(grads)
            proj_grad = self._project(grad_vector)
            test_grads.append(proj_grad.cpu())
        
        test_grads = torch.stack(test_grads).to(self.device)
        
        # Step 3: Compute influences
        print("\nStep 3/3: Computing influence scores...")
        for j in tqdm(range(n_test), desc="Computing influences"):
            test_grad_j = test_grads[j]
            
            for i in range(n_train):
                train_grad_i = train_grads[i]
                influence = -torch.dot(train_grad_i, test_grad_j).item()
                influences[i, j] = influence
        
        print("\n[VIF-CG] Attribution complete!")
        return influences


def get_vif_cg_scores(model, train_loader, val_loader, device='cuda', 
                      proj_dim=512, cg_max_iter=100):
    """Wrapper function for VIF-CG."""
    attributor = VIF_CG_Attributor(
        model=model,
        device=device,
        proj_dim=proj_dim,
        regularization=1e-5,
        cg_max_iter=cg_max_iter,
        cg_tol=1e-5
    )
    
    influences = attributor.attribute(
        train_loader=train_loader,
        test_loader=val_loader
    )
    
    return influences


def collate_with_index(batch):
    """Collate function that preserves indices."""
    images, captions, indices = zip(*batch)
    images = torch.stack(images)
    captions = torch.stack(captions)
    indices = torch.stack(indices)
    return ((images, captions), indices)


def build_tensor_dataset(coco_dataset, tokenizer, transform, N):
    """Build tensor dataset with indices."""
    image_tensor_list = []
    caption_tensor_list = []
    index_tensor_list = []

    for i in tqdm(range(N), desc="Preprocessing dataset"):
        img, captions = coco_dataset[i]
        
        img_tensor = transform(img)
        image_tensor_list.append(img_tensor)
        
        text_tensor = tokenizer(captions[0])
        caption_tensor_list.append(text_tensor)
        
        index_tensor_list.append(torch.tensor(i))

    image_tensor_stack = torch.stack(image_tensor_list)
    caption_tensor_stack = torch.stack(caption_tensor_list)
    index_tensor_stack = torch.stack(index_tensor_list)

    print(f"\n✅ Dataset created:")
    print(f"  Images: {image_tensor_stack.shape}")
    print(f"  Captions: {caption_tensor_stack.shape}")
    print(f"  Indices: {index_tensor_stack.shape}")

    return TensorDataset(image_tensor_stack, caption_tensor_stack, index_tensor_stack)


def show_test_and_topk_train(
    influences, target_idx, 
    train_subset, train_dataset_raw_full,
    val_subset, val_dataset_raw_full,
    k=5,
    save_path=None,
    show=True
):
    """Visualize top-k influential training samples for a test sample."""
    val_raw_idx = val_subset.indices[target_idx]
    test_img, test_caps = val_dataset_raw_full[val_raw_idx]

    col = influences[:, target_idx]

    vals_top, idxs_top = torch.topk(torch.from_numpy(col), k=k)
    vals_bot, idxs_bot = torch.topk(-torch.from_numpy(col), k=k)

    plt.figure(figsize=(4*(k+1), 8))

    # Top row: Most positive influence
    ax = plt.subplot(2, k+1, 1)
    ax.imshow(test_img)
    ax.set_title(f"Test #{target_idx}\n{test_caps[0][:50]}", fontsize=9)
    ax.axis("off")

    for j, (i, v) in enumerate(zip(idxs_top.tolist(), vals_top.tolist()), 2):
        raw_idx = train_subset.indices[i]
        img, caps = train_dataset_raw_full[raw_idx]
        ax = plt.subplot(2, k+1, j)
        ax.imshow(img)
        ax.set_title(f"Top {j-1}: Train#{i}\nScore: {v:.4f}", fontsize=9)
        ax.axis("off")

    # Bottom row: Most negative influence
    ax = plt.subplot(2, k+1, k+2)
    ax.imshow(test_img)
    ax.set_title(f"Test #{target_idx}\n{test_caps[0][:50]}", fontsize=9)
    ax.axis("off")

    for j, (i, v) in enumerate(zip(idxs_bot.tolist(), (-vals_bot).tolist()), 2):
        raw_idx = train_subset.indices[i]
        img, caps = train_dataset_raw_full[raw_idx]
        ax = plt.subplot(2, k+1, k+1+j)
        ax.imshow(img)
        ax.set_title(f"Bot {j-1}: Train#{i}\nScore: {v:.4f}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def save_topk_for_all_tests(
    influences,
    train_subset,
    train_dataset_raw_full,
    val_subset,
    val_dataset_raw_full,
    k=5,
    out_dir="results",
    filename_tpl="top{K}_{idx}.png",
    show=False,
    indices=None
):
    """Save top-k visualizations for all test samples."""
    os.makedirs(out_dir, exist_ok=True)

    n_val = influences.shape[1]
    
    if indices is None:
        indices = list(range(n_val))

    for idx in tqdm(indices, desc=f"Saving top-{k} figures"):
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
            print(f"[warn] Failed on test idx={idx}: {e}")


# ==================== Main Script ====================

if __name__ == "__main__":
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP model and tokenizer
    model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer("RN50")
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4815, 0.4578, 0.4082),
                           std=(0.2686, 0.2613, 0.2758))
    ])
    
    # Load COCO datasets
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
    
    # Sample subsets
    random.seed(42)
    train_indices = random.sample(range(len(train_dataset_full)), 5000)
    val_indices = random.sample(range(len(val_dataset_full)), 500)
    
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    train_dataset_raw = Subset(train_dataset_raw_full, train_indices)
    val_dataset_raw = Subset(val_dataset_raw_full, val_indices)
    
    # Build tensor datasets
    N_train = len(train_dataset_raw)
    N_val = len(val_dataset_raw)
    
    print(f"\nBuilding tensor datasets...")
    train_tensor_dataset = build_tensor_dataset(
        train_dataset_raw, tokenizer, transform_train, N_train
    )
    val_tensor_dataset = build_tensor_dataset(
        val_dataset_raw, tokenizer, transform_train, N_val
    )
    
    # Create dataloaders
    loader_train_subset_raw = DataLoader(
        train_tensor_dataset, 
        batch_size=6, 
        shuffle=False,
        collate_fn=collate_with_index
    )
    loader_val_subset_raw = DataLoader(
        val_tensor_dataset, 
        batch_size=6, 
        shuffle=False,
        collate_fn=collate_with_index
    )
    
    # Load your trained model here
    full_model = model  # Replace with your trained model
    full_model.eval()
    
    print("\n" + "="*70)
    print("Starting VIF-CG Attribution")
    print("="*70)
    
    # ========== VIF-CG Attribution ==========
    influences = get_vif_cg_scores(
        model=full_model,
        train_loader=loader_train_subset_raw,
        val_loader=loader_val_subset_raw,
        device=device,
        proj_dim=512,      
        cg_max_iter=100     
    )
    
    influences = torch.as_tensor(influences)
    print(f"\n Influence matrix shape: {influences.shape}")
    
    # ========== Visualization ==========
    print("\nGenerating visualizations...")
    save_topk_for_all_tests(
        influences=influences,
        train_subset=train_dataset,
        train_dataset_raw_full=train_dataset_raw_full,
        val_subset=val_dataset,
        val_dataset_raw_full=val_dataset_raw_full,
        k=5,
        out_dir="results_vif",
        filename_tpl="top{K}_{idx}.png",
        show=False
    )
    
    print("\n All done! Check 'results_vif/' for output images.")