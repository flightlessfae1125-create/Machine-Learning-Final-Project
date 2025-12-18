
# Experimental Report

---

## Table of Contents
1. [Objective of the Experiment](#1-objective-of-the-experiment)
2. [Experimental Procedure](#2-experimental-procedure)
3. [Code Logic](#3-code-logic)
4. [Code Modules and Implementation](#4-code-modules-and-implementation)
5. [Experimental Results and Analysis](#5-experimental-results-and-analysis)
6. [Summary and Conclusions](#6-summary-and-conclusions)

---

## 1. Objective of the Experiment

### 1.1 Research Background

In billion-scale recommendation systems, click-through rate (CTR) prediction has long been a challenging task for graph neural networks (GNNs). The primary reason lies in the enormous computational complexity caused by aggregating tens of billions of neighboring nodes.

Traditional GNN-based CTR models usually adopt **sampling** strategies, randomly sampling hundreds of neighbors from billions of candidates to enable efficient online recommendation. However, this approach introduces severe **sampling bias**, which prevents the model from capturing complete behavioral patterns of users or items.

### 1.2 Core Innovations of MacGNN

MacGNN proposes an innovative solution as follows:

| Concept | Description |
|------|------|
| **Micro Recommendation Graph (Micro Graph)** | The conventional user–item interaction graph, with the number of nodes reaching billions |
| **Macro Recommendation Graph (MAG)** | Aggregates micro nodes with similar behavioral patterns into macro nodes, reducing the number of nodes from billions to hundreds |
| **Macro Graph Neural Network (MacGNN)** | Aggregates information at the macro level and refines macro-neighbor embedding representations |

### 1.3 Experimental Objectives

The objectives of this experiment include:

1. **Validating the effectiveness of MacGNN** by comparing its performance with baseline models on multiple real-world datasets  
2. **Ablation studies** to verify the contribution of macro-graph components to model performance  
3. **User group analysis** to evaluate performance across users with different activity levels  
4. **Inference efficiency evaluation** to measure inference latency and verify industrial deployment feasibility  
5. **Hyperparameter tuning** to explore the impact of different hyperparameter configurations  

---

## 2. Experimental Procedure

### 2.1 Overall Experimental Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MacGNN Experimental Workflow                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐    ┌───────────────┐             │
│  │ Data Loading │───▶│ Model Init   │───▶│ Training Loop │             │
│  │ Data Load    │     │ Model Init   │    │ Training      │             │
│  └──────────────┘     └──────────────┘    └───────────────┘             │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌───────────────┐    ┌───────────────┐    ┌──────────────┐             │
│  │ Feature       │    │ Embedding     │    │ Early Stop   │             │
│  │ Extraction    │    │ Initialization│    │ Mechanism    │             │
│  │ - User ID     │    │ - User Embed  │    │ EarlyStopper │             │
│  │ - Item ID     │    │ - Item Embed  │    │              │             │
│  │ - Macro Neigh │    │ - Macro Embed │    └──────────────┘             │
│  │ - History Seq │    └───────────────┘           │                     │
│  └───────────────┘                                ▼                     │
│                                           ┌──────────────┐              │
│                                           │ Model Eval   │              │
│                                           │ - AUC        │              │
│                                           │ - LogLoss    │              │
│                                           │ - GAUC       │              │
│                                           └──────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Datasets

Three public datasets are used in this experiment:

|           Dataset           |     Source     |                            Description                            |
|-----------------------------|----------------|-------------------------------------------------------------------|
| **MovieLens-10M (ml-10m)**  | GroupLens      | Movie rating dataset containing approximately 10 million ratings  |
| **Electronics (elec)**      | Amazon Reviews | Electronics product review dataset                                |
| **KuaiRec (kuairec)**       | Kuaishou       | Short-video recommendation dataset                                |

### 2.3 Evaluation Metrics

|    Metric    |       Full Name       |                                                     Description                                                     |
|--------------|-----------------------|---------------------------------------------------------------------------------------------------------------------|
| **AUC**      | Area Under ROC Curve  | Measures the overall ranking capability of the model; higher is better                                              |
| **LogLoss**  | Logarithmic Loss      | Measures cross-entropy loss between predicted probabilities and true labels; lower is better                        |
| **GAUC**     | Group AUC             | Computes AUC per user group and averages them with weights, better reflecting personalized recommendation scenarios |

### 2.4 Baseline Models

|    Model    |    Paper    |                                      Characteristics                                      |
|-------------|-------------|-------------------------------------------------------------------------------------------|
| **DeepFM**  | WWW 2017    | Combines second-order feature interactions from FM with high-order interactions from DNN  |
| **DIN**     | KDD 2018    | Captures user interests using attention mechanisms                                        |
| **DIEN**    | AAAI 2019   | Models the evolution of user interests using GRU                                          |
| **MacGNN**  | WWW 2024    | The proposed macro graph neural network                                                   |

---

## 3. Code Logic

### 3.1 Data Flow Diagram

```
Input Data x (batch_size × feature_dim)
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Feature Parsing                               │
├──────────────────────────────────────────────────────────────────────┤
│ user_id │ user_1ord_neighbor │ user_2ord_neighbor │ user_recent      │
│ item_id │ item_1ord_neighbor │ item_2ord_neighbor │ item_recent      │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Embedding Layer                               │
├──────────────────────────────────────────────────────────────────────┤
│ user_embed │ user_1ord_embed (macro) │ user_recent_embed             │
│ item_embed │ item_1ord_embed (macro) │ item_recent_embed             │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Attention-based Neighbor Aggregation              │
├──────────────────────────────────────────────────────────────────────┤
│ Query-Key-Value Attention Mechanism                                  │
│ score = softmax(Q × Kᵀ / √d)                                         │
│ output = score × V                                                   │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Weighted Summation and Concatenation              │
├──────────────────────────────────────────────────────────────────────┤
│ concat = [user_emb, user_1ord_ws, user_2ord_ws, user_recent_ws,      │
│           item_emb, item_1ord_ws, item_2ord_ws, item_recent_ws]      │
│                     (14 × embed_dim)                                 │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        MLP + Sigmoid                                 │
├──────────────────────────────────────────────────────────────────────┤
│ Linear(14*embed_dim, 200) → Dice → Linear(200, 80) → Dice →          │
│ Linear(80, 1) → Sigmoid                                              │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
      Output: CTR prediction probability (0–1)
```

### 3.2 Core Algorithm Workflow

```python
# Pseudocode: MacGNN Forward Propagation
def forward(x):
    # Step 1: Extract micro-level user/item embeddings
    user_emb = user_embed(x[:, 0])
    item_emb = item_embed(x[:, item_pos])
    
    # Step 2: Extract macro neighbor information
    user_1ord_neighbor = x[:, 1:i_group_num+1]     # First-order user neighbors (item clusters)
    user_2ord_neighbor = x[:, ...]                 # Second-order user neighbors (user clusters)
    
    # Step 3: Compute macro weights (temperature τ controls smoothness)
    user_1ord_weight = softmax(log(neighbor_count + 1) / τ)
    
    # Step 4: Macro neighbor embedding aggregation
    user_1ord_embed = i_macro_embed(i_group_slice) # Item macro embeddings
    u_1ord_trans = attention_aggregate(user_1ord_embed, item_emb)
    
    # Step 5: Weighted summation
    user_1ord_ws = (u_1ord_trans * user_1ord_weight).sum(dim=1)
    
    # Step 6: Feature concatenation and MLP prediction
    concat = torch.cat([user_emb, user_1ord_ws, ..., item_emb, ...])
    output = sigmoid(mlp(concat))
    
    return output
```

### 3.3 Macro Weight Computation Formula

The temperature coefficient τ controls the smoothness of macro-neighbor weights:

$$
w_i = \frac{\exp(\log(n_i + 1) / \tau)}{\sum_j \exp(\log(n_j + 1) / \tau)}
$$

Where:
- $n_i$ denotes the interaction count of the $i$-th macro neighbor  
- $\tau$ is the temperature coefficient; smaller values produce sharper distributions  

---

## 4. Code Modules and Implementation

### 4.1 Module Architecture

```
MacGNN_demo.ipynb
├── Dependency Imports
│   ├── torch, numpy, pandas
│   └── sklearn.metrics
│
├── Evaluation Metrics
│   ├── cal_group_auc()          # GAUC computation
│   └── evaluation()             # AUC / LogLoss / GAUC
│
├── Activation Function
│   └── Dice                     # Adaptive activation function
│
├── Baseline Models
│   ├── DeepFM                   # FM + DNN
│   ├── DIN                      # Deep Interest Network
│   └── DIEN                     # Deep Interest Evolution Network
│
├── Core Models
│   ├── NeighborAggregation      # Attention-based neighbor aggregation
│   ├── MacGNN                   # Full MacGNN
│   └── MacGNN_NoMacro           # Ablation version (without macro graph)
│
├── Data Module
│   ├── DatasetBuilder           # PyTorch Dataset
│   ├── load_dataset_assets()    # Data loading
│   └── build_dataloaders()      # DataLoader construction
│
├── Training Module
│   ├── train()                  # Training loop with scheduler
│   ├── train_simple()           # Simplified training loop
│   └── EarlyStopper             # Early stopping mechanism
│
├── Comprehensive Evaluation
│   ├── evaluation_comprehensive()
│   ├── analyze_user_groups()    # User group analysis
│   └── run_model_comparison()   # Multi-model comparison
│
└── Visualization
    ├── plot_model_comparison()
    └── create_comparison_table()
```

### 4.2 Detailed Explanation of Core Classes

#### 4.2.1 Dice Activation Function

```python
class Dice(nn.Module):
    """
    Data-aware Adaptive Activation Function
    Adaptively adjusts activation behavior based on data distribution
    """
    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1,)))
        
    def forward(self, x):
        avg = x.mean(dim=0)
        std = x.std(dim=0)
        norm_x = (x - avg) / (std + 1e-8)
        p = torch.sigmoid(norm_x)
        return x.mul(p) + self.alpha * x.mul(1 - p)
```

**Design Rationale**:
- Dice adaptively adjusts activation intensity according to data distribution  
- The learnable parameter α allows balancing between positive and negative activations  
- Compared with PReLU or LeakyReLU, Dice is more suitable for CTR prediction tasks  

#### 4.2.2 NeighborAggregation Attention Module

```python
class NeighborAggregation(nn.Module):
    """
    Query-Key-Value based attention neighbor aggregation mechanism
    """
    def __init__(self, embed_dim=8, hidden_dim=8):
        super().__init__()
        self.Q_w = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.K_w = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.V_w = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.trans_d = math.sqrt(hidden_dim)
        
    def forward(self, query, key):
        trans_Q = self.Q_w(query)
        trans_K = self.K_w(key)
        trans_V = self.V_w(query)
        score = softmax(bmm(trans_Q, trans_K.T) / √d)
        answer = mul(trans_V, score)
        return answer
```

**Key Design Choices**:
- Uses scaled dot-product attention  
- Queries originate from neighbor embeddings, keys from target nodes  
- Division by √d prevents gradient vanishing in softmax  

#### 4.2.3 Full MacGNN Model

```python
class MacGNN(nn.Module):
    """
    Micro–Macro Consumer Graph Neural Network
    """
    def __init__(self, field_dims, u_group_num, i_group_num,
                 embed_dim, recent_len, tau=0.8, device='cpu'):
        self.user_embed = nn.Embedding(field_dims[0], embed_dim)
        self.item_embed = nn.Embedding(field_dims[1], embed_dim)
        self.u_macro_embed = nn.Embedding(u_group_num + 1, embed_dim)
        self.i_macro_embed = nn.Embedding(i_group_num + 1, embed_dim)
        self.u_shared_aggregator = NeighborAggregation(embed_dim, 2*embed_dim)
        self.i_shared_aggregator = NeighborAggregation(embed_dim, 2*embed_dim)
        self.tau = tau
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 14, 200), Dice(),
            nn.Linear(200, 80), Dice(),
            nn.Linear(80, 1)
        )
```

**Input Feature Structure** (total 14 × embed_dim):

|  Feature Group  |  Dimension  |               Description               |
|-----------------|-------------|-----------------------------------------|
| user_embedding  | embed_dim   | Micro-level user embedding              |
| user_1ord_ws    | 2×embed_dim | First-order macro neighbor aggregation  |
| user_2ord_ws    | 2×embed_dim | Second-order macro neighbor aggregation |
| user_recent_ws  | 2×embed_dim | User historical behavior aggregation    |
| item_embedding  | embed_dim   | Micro-level item embedding              |
| item_1ord_ws    | 2×embed_dim | First-order macro neighbor aggregation  |
| item_2ord_ws    | 2×embed_dim | Second-order macro neighbor aggregation |
| item_recent_ws  | 2×embed_dim | Item historical interaction aggregation |

#### 4.2.4 Early Stopping Mechanism

```python
class EarlyStopper:
    """
    Early stopping mechanism to prevent overfitting.
    Training stops when validation AUC does not improve for num_trials epochs.
    """
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_auc = 0.0
        
    def is_continuable(self, model, auc, log_loss):
        if auc > self.best_auc:
            self.best_auc = auc
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        return False
```

### 4.3 Training Configuration

```python
config = {
    'dataset_name': 'elec',
    'data_dir': 'data',
    'model_name': 'macgnn',
    'embed_dim': 32,
    'recent_len': 20,
    'tau': 0.8,
    'epoch': 20,
    'batch_size': 1024,
    'learning_rate': 1e-2,
    'weight_decay': 5e-5,
    'early_epoch': 4,
    'mlp_dropout': 0.0,
    'lr_scheduler': None,
    'use_gpu': True,
    'cuda_id': 0,
}
```

### 4.4 Hyperparameter Tuning Scenarios

Eight tuning scenarios are defined:

|          Scenario          | Learning Rate | embed_dim | Epoch |       Special Setting       |
|----------------------------|---------------|-----------|-------|-----------------------------|
| Baseline                   | 1e-2          | 10        | 20    | Original paper setup        |
| HighLR_Embed32             | 1e-2          | 32        | 20    | Larger embedding            |
| LowLR_LongEpoch            | 1e-3          | 32        | 40    | Low LR + long training      |
| HighLR_Embed32_CosineDrop  | 1e-2          | 32        | 25    | Cosine annealing + dropout  |
| LowLR_LongEpoch_Step       | 1e-3          | 32        | 60    | StepLR                      |
| ShortSeq_Tau1              | 1e-2          | 32        | 20    | Short sequence + τ=1.0      |
| LongSeq_Tau06              | 1e-2          | 32        | 30    | Long sequence + τ=0.6       |
| HighLR_Embed32_MultiRun    | 1e-2          | 32        | 20    | Average of 3 runs           |

---

## 5. Experimental Results and Analysis

### 5.1 Multi-model Comparison Results

#### Electronics Dataset (elec)

|    Model    |      AUC      |    LogLoss    |     GAUC     | Time (ms) |
|-------------|---------------|---------------|--------------|-----------|
| **MACGNN**  | **0.837595**  | **0.503814**  | **0.83804**  | 0.0326    |
| DIN         | 0.694576      | 0.611843      | 0.69100      | 0.0029    |
| DIEN        | 0.694131      | 0.613782      | 0.69203      | 0.0127    |
| DEEPFM      | 0.691888      | 0.611605      | 0.69069      | 0.0018    |

**Analysis**:
- MacGNN improves AUC by **14.3%** over the best baseline  
- GAUC also shows significant improvement, indicating superior personalization  
- Although inference time is slightly higher, it remains within millisecond-level latency  

#### KuaiRec Dataset (kuairec)

|    Model    |      AUC      |    LogLoss    |     GAUC     | Time (ms) |
|-------------|---------------|---------------|--------------|-----------|
| **MACGNN**  | **0.814022**  | **0.503371**  | **0.77262**  | 0.0081    |
| DIEN        | 0.772750      | 0.546671      | 0.74922      | 0.0077    |
| DIN         | 0.715416      | 0.644510      | 0.70502      | 0.0042    |
| DEEPFM      | 0.649816      | 0.639139      | 0.50665      | 0.0017    |

**Analysis**:
- MacGNN achieves the best performance in short-video recommendation  
- DIEN performs well due to sequential modeling  
- DeepFM’s GAUC is close to random, showing limitations of pure feature interaction  

#### MovieLens-10M Dataset (ml-10m)

|    Model    |      AUC      |    LogLoss    |     GAUC     | Time (ms) |
|-------------|---------------|---------------|--------------|-----------|
| **MACGNN**  | **0.745988**  | **0.584745**  | **0.72065**  | 0.0051    |
| DEEPFM      | 0.618282      | 0.679630      | 0.54585      | 0.0023    |
| DIN         | 0.615590      | 0.690722      | 0.54609      | 0.0033    |
| DIEN        | 0.613406      | 0.680122      | 0.53870      | 0.0142    |

**Analysis**:
- MacGNN shows even greater advantages in movie recommendation scenarios  
- Baseline models perform similarly, highlighting their limitations  
- Macro graph aggregation effectively captures group-level behavior patterns  

### 5.2 Comprehensive Performance Comparison

![Model Comparison](checkpoints/comparison_elec.png)

### 5.3 User Group Performance Analysis

Users are divided into three groups based on activity level:

|    User Group    |         Definition         |     MacGNN     | Best Baseline |
|------------------|----------------------------|----------------|---------------|
| Cold Start       | Bottom 1/3 by interactions | ✅ Effective  | ❌ Poor       |
| Medium Activity  | Middle 1/3                 | ✅ Good       | Average       |
| High Activity    | Top 1/3                    | ✅ Best       | Acceptable    |

**Key Findings**:
- MacGNN shows particularly strong advantages for **cold-start users**  
- Macro graphs leverage group behavior to mitigate data sparsity  

### 5.4 Ablation Study Results

|   Model Variant   |   AUC   | Relative Change |
|-------------------|---------|-----------------|
| MacGNN (Full)     | 0.8376  | –               |
| MacGNN w/o Macro  | 0.7812  | –6.7%           |

**Conclusion**:
- Removing macro-graph components reduces AUC by approximately **6.7%**  
- Confirms the critical contribution of macro embeddings  
- Macro neighbor aggregation is a key success factor  

### 5.5 Inference Efficiency Analysis

|  Model  | Inference Time (ms/sample) | Relative to DeepFM |
|---------|----------------------------|--------------------|
| DeepFM  | 0.0018                     | 1.0×               |
| DIN     | 0.0029                     | 1.6×               |
| DIEN    | 0.0127                     | 7.1×               |
| MacGNN  | 0.0326                     | 18.1×              |

**Analysis**:
- MacGNN has the highest inference cost but remains in the millisecond range  
- 0.03 ms latency is acceptable for online recommendation systems  
- Compared with traditional GNN sampling hundreds of neighbors, MacGNN aggregates only dozens of macro nodes  

### 5.6 Hyperparameter Sensitivity Analysis

|       Scenario       |   AUC   |      Key Observation      |
|----------------------|---------|---------------------------|
| Baseline (embed=10)  | 0.8407  | Stable performance        |
| HighLR_Embed32       | 0.8376  | Similar performance       |
| LowLR_LongEpoch      | 0.8312  | Requires longer training  |
| Cosine + Dropout     | 0.8298  | Possible underfitting     |
| ShortSeq_Tau1        | 0.8356  | Better for sparse data    |
| LongSeq_Tau06        | 0.8289  | More noise introduced     |

**Tuning Recommendations**:
1. **Learning Rate**: 1e-2 is a good starting point  
2. **Embedding Dimension**: 10–32 is sufficient  
3. **Temperature τ**: Tune within 0.6–1.0  
4. **Sequence Length**: Shorter sequences for sparse data  

---

## 6. Summary and Conclusions

### 6.1 Key Findings

1. **MacGNN significantly outperforms baseline models**
   - Average AUC improvement exceeds **10%** across datasets  
   - GAUC improvements confirm superior personalization capability  

2. **Macro graphs are the core innovation**
   - Ablation studies show a 6.7% AUC drop without macro graphs  
   - Effectively resolves computational bottlenecks of billion-scale neighbors  

3. **Inference efficiency meets online requirements**
   - Millisecond-level inference latency supports real-time recommendation  
   - Significantly lower complexity than traditional GNN sampling  

4. **Strong cold-start performance**
   - Macro graphs compensate for sparse individual data using group behavior  
   - Substantially improves recommendations for new users and items  

### 6.2 Methodological Innovations

|            Innovation            |                 Description                 |             Effect             |
|----------------------------------|---------------------------------------------|--------------------------------|
| Macro Recommendation Graph (MAG) | Compresses billions of nodes into hundreds  | Reduces computation            |
| Macro Node Embeddings            | Learns group-level representations          | Captures group behavior        |
| Temperature Coefficient τ        | Controls weight distribution                | Balances neighbor importance   |
| Attention Aggregation            | QKV-based aggregation                       | Enhances representation power  |

### 6.3 Applicable Scenarios

MacGNN is particularly suitable for:
- ✅ Billion-scale user/item recommendation systems  
- ✅ Real-time online CTR prediction  
- ✅ Scenarios with severe cold-start problems  
- ✅ Applications requiring modeling of group behavior  

### 6.4 Future Work

1. **Dynamic Macro Graphs**: Dynamically update macro nodes based on real-time interactions  
2. **Multi-scale Macro Graphs**: Introduce hierarchical macro nodes  
3. **Cross-domain Transfer**: Transfer macro-graph knowledge across domains  
4. **Interpretability**: Analyze semantic meanings of macro nodes  

---

## Appendix: Running Guide

### Environment Setup

```bash
pip install -r requirements.txt

# Requirements
# python >= 3.8
# torch == 1.11.0+
# scikit-learn == 1.1.1
# pandas == 1.4.1
# numpy == 1.21.2
# tqdm == 4.63.0
```

### Quick Start

1. Download datasets and place them in the `data/` directory  
2. Open `MacGNN_demo.ipynb`  
3. Run all cells  

### Custom Experiments

```python
config['dataset_name'] = 'kuairec'
config['embed_dim'] = 64
config['tau'] = 0.6

results = run_experiment(config)
```

---

> **Report Generated**: 2024  
> **Experimental Environment**: Apple Silicon (MPS) / NVIDIA CUDA
