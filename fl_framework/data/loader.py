# fl_framework/data/loader.py
import numpy as np
from torch.utils.data import Subset
from typing import List
from .base_dataset import BaseFLDataset
import sys

class NonIIDSubsetWapper(Subset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.true_len = super().__len__()

    def __len__(self):
        return self.true_len

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i % self.true_len] for i in idx]]
        return self.dataset[self.indices[idx % self.true_len]]

    def __getitems__(self, indices: List[int]):
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx % self.true_len] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.dataset[self.indices[idx % self.true_len]] for idx in indices]
    
    
            


def partition_dataset(
    dataset: BaseFLDataset,
    num_clients: int,
    iid: bool = True,
    seed: int = 42,
    alpha: int = 0.2 # Only used for Non-IID
) -> List[Subset]:
    """
    Partitions a given dataset object among multiple clients.

    This function is decoupled from dataset creation. It operates on any
    dataset that conforms to the BaseFLDataset interface.

    Args:
        dataset (BaseFLDataset): The dataset instance to partition.
        num_clients (int): The total number of clients.
        iid (bool): If True, partitions data in an IID fashion.
                    If False, partitions data in a Non-IID fashion by label.
        seed (int): Random seed for reproducibility.
        shards_per_client (int): How many label shards each client gets in Non-IID mode.

    Returns:
        List[Subset]: A list of Subset objects, one for each client.
    """
    if iid:
        partitions = _partition_iid(dataset, num_clients, seed)
    else:
        # partitions = _partition_non_iid_by_label(
        #     dataset, num_clients, seed, shards_per_client
        # )
        partitions = _partition_non_iid_dirichlet(
            dataset,
            num_clients,
            seed=seed,
            alpha=alpha
        )
    
    print(f"Partitioned dataset into {len(partitions)} {'IID' if iid else 'Non-IID'} subsets.")
    return partitions


def _partition_iid(dataset: BaseFLDataset, num_clients: int, seed: int) -> List[Subset]:
    """Partitions a dataset into IID subsets."""
    num_items_per_client = len(dataset) // num_clients
    client_indices = [[] for _ in range(num_clients)]
    all_indices = list(range(len(dataset)))

    rng = np.random.default_rng(seed)
    rng.shuffle(all_indices)

    for i in range(num_clients):
        start_idx = i * num_items_per_client
        end_idx = (i + 1) * num_items_per_client
        client_indices[i] = all_indices[start_idx:end_idx]

    return [Subset(dataset, indices) for indices in client_indices]

def _partition_non_iid_dirichlet(
    dataset: BaseFLDataset, 
    num_clients: int, 
    alpha: float, 
    seed: int) -> List[Subset]:


    """
    Partitions a dataset into Non-IID subsets using Dirichlet distribution.
    
    Args:
        dataset: The source dataset.
        num_clients: Number of clients.
        alpha: Concentration parameter for Dirichlet distribution. 
               Smaller alpha means higher heterogeneity (non-IID).
               Larger alpha means closer to IID.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    
    # 1. 获取数据集的标签 (Labels/Targets)
    y = np.array(dataset.targets)
        
    # 确保标签是整数类型
    y = y.astype(int)
    num_classes = len(np.unique(y))
    N = len(dataset)
    
    # 存储每个客户端分配到的索引
    client_indices = [[] for _ in range(num_clients)]
    
    # 2. 对每个类别进行 Dirichlet 分配
    for k in range(num_classes):
        # 获取属于类别 k 的所有样本索引
        idx_k = np.where(y == k)[0]
        rng.shuffle(idx_k) # 打乱该类别的索引
        
        # 生成 Dirichlet 分布比例
        # proportions 长度为 num_clients，和为 1
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        
        # 抑制已超过平均分配量的客户端继续获取数据
        # 使用 np.maximum 保底，避免所有比例归零导致除以零
        proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, client_indices)])
        total = proportions.sum()
        if total == 0:
            # 所有客户端均已超过均值，退回原始 Dirichlet 比例
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        else:
            proportions = proportions / total
        
        # 计算分割点
        # (cumsum * len) 得到累积的分割索引位置
        split_idx = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        # 将该类别的索引切分给各个客户端
        idx_k_split = np.split(idx_k, split_idx)
        
        # 将切分后的索引追加到对应客户端列表
        for i in range(num_clients):
            client_indices[i].extend(idx_k_split[i])

    # 保证非空 (Guarantee Non-Empty)
    # 统计当前每个客户端的样本数
    lengths = [len(indices) for indices in client_indices]
    
    # 只要存在样本数为 0 的客户端，就执行循环
    while np.min(lengths) == 0:
        # 1. 找出所有空的客户端索引
        empty_indices = np.where(np.array(lengths) == 0)[0]
        
        for empty_idx in empty_indices:
            # 2. 找出当前最富有的客户端（样本最多）
            wealthy_idx = np.argmax(lengths)
            
            # 3. 如果最富有的客户端也没得借（理论上被 len(dataset) check 排除，但在循环中需防备）
            if lengths[wealthy_idx] < 2:
                break # 无法再平衡
            
            # 4. 移动一个样本：从富有者 pop，给贫穷者 append
            # 因为之前已经 shuffle 过或按类分配，pop 最后一个样本是随机类别的
            sample_to_move = client_indices[wealthy_idx].pop()
            client_indices[empty_idx].append(sample_to_move)
            
            # 5. 更新长度记录，以便下一次循环判断
            lengths[wealthy_idx] -= 1
            lengths[empty_idx] += 1
            
            # 如果最富有的客户端已经不是最富有的了，或者空的已经填满，
            # 实际上我们在下一次 while 循环会重新计算，这里只是为了当前 for 循环的准确性

    # 打乱每个客户端内部的索引顺序
    for i in range(num_clients):
        rng.shuffle(client_indices[i])
    
    return [NonIIDSubsetWapper(dataset, indices) for indices in client_indices]


# def _partition_non_iid_by_label(
#     dataset: BaseFLDataset,
#     num_clients: int,
#     seed: int,
#     shards_per_client: int
# ) -> List[Subset]:
#     """Partitions a dataset into Non-IID subsets based on label distribution."""
#     num_shards = num_clients * shards_per_client
#     if len(dataset) < num_shards:
#         raise ValueError("Not enough data for the specified number of shards.")
#     num_images_per_shard = len(dataset) // num_shards
    
#     # Efficiently get labels thanks to the BaseFLDataset interface
#     labels = np.array(dataset.targets)
    
#     # 1. Sort data indices by their corresponding labels
#     indices_by_label = np.argsort(labels)
    
#     # 2. Create shards from the sorted data
#     shards = []
#     for i in range(num_shards):
#         start = i * num_images_per_shard
#         end = start + num_images_per_shard
#         shards.append(indices_by_label[start:end])

#     # 3. Assign shards to clients randomly
#     rng = np.random.default_rng(seed)
#     rng.shuffle(shards)
    
#     client_indices = [[] for _ in range(num_clients)]
#     for i in range(num_clients):
#         start_shard_idx = i * shards_per_client
#         end_shard_idx = (i + 1) * shards_per_client
#         assigned_shards = shards[start_shard_idx:end_shard_idx]
#         client_indices[i] = np.concatenate(assigned_shards).tolist()

#     return [Subset(dataset, indices) for indices in client_indices]
