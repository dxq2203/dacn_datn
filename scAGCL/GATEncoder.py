# This file is implemented based on this repository: https://github.com/gordicaleksa/pytorch-GAT/blob/main/models/definitions/GAT.py

import torch
import torch.nn as nn 
import torch.nn.functional as F

class GATLayer(nn.Module):
    src_nodes_dim = 0 # position of source nodes in edge index
    trg_nodes_dim = 1 # position of target node in edge index

    nodes_dim = 0 # node dimension/axis
    head_dim = 1 # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
        dropout_prob=0.6, add_skip_connection=True, bias=True):
        
        super().__init__()

        # self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.num_of_heads = num_of_heads
        self.concat = concat
        # self.activation = activation
        # self.dropout_prob = dropout_prob
        self.add_skip_connection = add_skip_connection
        # self.bias = bias

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # Linear projection matrix W
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        # Bias
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leaky_ReLU = nn.LeakyReLU(0.2)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_prob)

        self.init_params()

    
    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    
    def forward(self, data):
        # Note that data here is a tuple: (in_nodes_features, edge_index)
        in_nodes_features, edge_index = data    # unpack data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'
        

        #
        # Step 1: Linear projection
        #

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)     # in the official GAT imp they did dropout here as well


        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Note: Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source_lifted = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target_lifted = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_proj_lifted = nodes_features_proj.index_select(self.nodes_dim, src_nodes_index)

        # Compute the attention coefficients for each edge
        
        # As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        # Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        # into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        # in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        # (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
        #  i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        # Note:
        # Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        # and it's a fairly common "trick" used in pretty much every deep learning framework.
        # Check out this link for more details:
        # https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        scores_per_edge = self.leaky_ReLU(scores_source_lifted + scores_target_lifted)

        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()

        # Calculate the denominator. Shape = (E, NH)
        neighborhood_sum_denominator = self.sum_edge_scores_neighborhood(exp_scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attention_per_edge = exp_scores_per_edge / (neighborhood_sum_denominator + 1e-16)
        attention_per_edge = attention_per_edge.unsqueeze(-1)
        attention_per_edge = self.dropout(attention_per_edge)


        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attention_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)


        #
        # Step 4: Residual/skip connections, concat and bias
        #

        if self.add_skip_connection: # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]: # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim = self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        if self.activation is not None:
            out_nodes_features = self.activation(out_nodes_features)

        return out_nodes_features, edge_index


    def sum_edge_scores_neighborhood(self, exp_scores_per_edge, trg_nodes_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_nodes_index_expanded = self.check_shape(trg_nodes_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sum = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sum.scatter_add_(self.nodes_dim, trg_nodes_index_expanded, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sum.index_select(self.nodes_dim, trg_nodes_index)
    

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_nodes_index_expanded = self.check_shape(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)

        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_nodes_index_expanded, nodes_features_proj_lifted_weighted)

        return out_nodes_features


    def check_shape(self, this, other):
        # Append singleton dimension until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)
    

class GATEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, dropout_prob):
        super().__init__()

        # Layer 1: GAT with multi-head, output is concatenated and activated by ELU
        self.gat_layer_1 = GATLayer(
            num_in_features= in_features,
            num_out_features= hidden_features,
            num_of_heads= num_heads,
            concat= True,
            activation= nn.ELU(),
            dropout_prob= dropout_prob
        )

        self.dropout = nn.Dropout(dropout_prob)

        # Layer 2: GAT with multi-head, output is averaged and no activation
        self.gat_layer_2 = GATLayer(
            num_in_features= hidden_features * num_heads,
            num_out_features= out_features,
            num_of_heads= 1,
            concat= False,
            activation= None,
            dropout_prob= dropout_prob
        )

    
    def forward(self, x, adj):
        if adj.is_sparse:
            edge_index = adj._indices()

        else:
            edge_index = adj.nonzero().t().contiguous()

        x, _ = self.gat_layer_1((x, edge_index))
        x = self.dropout(x)
        x, _ = self.gat_layer_2((x, edge_index))

        return x

from torch_geometric.nn import GATv2Conv

class GATv2Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_features, num_heads=4, dropout_prob=0.6):
        super(GATv2Encoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, dropout=dropout_prob)
        # Second GATv2Conv layer with concat=False to get desired output dimensions
        self.conv2 = GATv2Conv(hidden_channels * num_heads, out_features, heads=1, concat=False, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, adj):
        if adj.is_sparse:
            edge_index = adj._indices()

        else:
            edge_index = adj.nonzero().t().contiguous()

        # First GAT layer with ReLU and dropout
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        # Second GAT layer
        x = self.conv2(x, edge_index)
        return x

class GATConvDense(nn.Module):
    """
    Phiên bản của lớp GAT có thể hoạt động với ma trận kề dày (dense).
    Điều này cho phép gradient chảy ngược qua ma trận kề,
    cần thiết cho các cuộc tấn công đối nghịch vào cấu trúc đồ thị.
    """
    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, bias=True):
        super().__init__()

        self.num_out_features = num_out_features
        self.num_of_heads = num_of_heads
        self.concat = concat

        # Ma trận trọng số W
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # Vector chú ý a
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        self.leaky_ReLU = nn.LeakyReLU(0.2)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x shape: (N, num_in_features), adj shape: (N, N)
        num_nodes = x.shape[0]

        if adj.is_sparse:
            dense_adj = adj.to_dense()
        else:
            dense_adj = adj

        adj_with_self_loops = dense_adj + torch.eye(num_nodes, device=adj.device)
        # Clamp value to 0-1
        adj_with_self_loops = torch.clamp(adj_with_self_loops, 0, 1)

        # 1. Biến đổi đặc trưng tuyến tính
        nodes_features_proj = self.linear_proj(x).view(num_nodes, self.num_of_heads, self.num_out_features)
        if torch.isnan(nodes_features_proj).any(): 
          print("NaN in nodes_features_proj!")
        
        # 2. Tính toán điểm chú ý cho tất cả các cặp node
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        if torch.isnan(scores_source).any(): 
          print("NaN in scores_source!")
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        if torch.isnan(scores_target).any(): 
          print("NaN in scores_target!")
        e = self.leaky_ReLU(scores_target.unsqueeze(1) + scores_source.unsqueeze(0))
        if torch.isnan(e).any(): 
          print("NaN in e!")

        # e = torch.tanh(e)

        # 3. Áp dụng mặt nạ từ ma trận kề một cách "mềm" (khả vi)
        # THAY THẾ KHỐI LỆNH CŨ BẰNG KHỐI LỆNH MỚI NÀY
        # ------------------------------------------------------------------
        # Khối lệnh cũ bị lỗi:
        # adj_mask = dense_adj.unsqueeze(-1) > 0
        # attention_logits = torch.where(adj_mask, e, -9e15 * torch.ones_like(e))

        # Khối lệnh MỚI, khả vi:
        # Tạo một mặt nạ cộng (additive mask). 
        # Nơi nào không có cạnh (dense_adj == 0), chúng ta sẽ trừ đi một số rất lớn.
        # Nơi nào có cạnh (dense_adj > 0), chúng ta cộng 0.
        # Phép toán (1.0 - dense_adj) đảm bảo rằng gradient có thể chảy qua dense_adj.
        # zero_vec = -1e9 * torch.ones_like(e)

        additive_mask = (1.0 - adj_with_self_loops) * -1e9 
        attention_logits = e + additive_mask.unsqueeze(-1)

        # adj_mask = adj_with_self_loops.unsqueeze(-1).expand_as(e)
        # attention_logits = e.masked_fill(adj_mask == 0, -1e9)
        if torch.isnan(attention_logits).any(): 
          print("NaN in attention_logits! Logits min/max: ", attention_logits.min(), attention_logits.max())
        # ------------------------------------------------------------------
        # attention_logits = attention_logits - attention_logits.max(dim=1, keepdim=True)[0]
        # attention_logits = torch.nan_to_num(attention_logits, nan=-1e4, neginf=-1e4)

        # row_max = attention_logits.max(dim=1, keepdim=True)[0]
        # mask_disconnected = row_max < -100
        # attention_logits = torch.where(mask_disconnected, torch.zeros_like(attention_logits), attention_logits)

        # 4. Áp dụng softmax và dropout
        attention = F.softmax(attention_logits, dim=1)
        if torch.isnan(attention).any(): 
          print("NaN in attention! Logits min/max: ", attention_logits.min(), attention_logits.max())
        attention = self.dropout(attention) # (N, N, NH)

        # 5. Tổng hợp đặc trưng từ hàng xóm
        attention_permuted = attention.permute(2, 0, 1)
        nodes_features_proj_permuted = nodes_features_proj.permute(1, 0, 2)
        out_nodes_features_permuted = torch.bmm(attention_permuted, nodes_features_proj_permuted)
        out_nodes_features = out_nodes_features_permuted.permute(1, 0, 2)

        # 6. Nối hoặc lấy trung bình các head
        if self.concat:
            out_nodes_features = out_nodes_features.reshape(num_nodes, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=1)

        if self.bias is not None:
            out_nodes_features += self.bias

        if self.activation is not None:
            out_nodes_features = self.activation(out_nodes_features)
        # print("out_nodes_features", out_nodes_features)
        return out_nodes_features

class GATEncoderDense(nn.Module):
    """
    Bộ mã hóa GAT hai lớp sử dụng GATConvDense.
    """
    def __init__(self, in_features, hidden_features, out_features, num_heads, dropout_prob):
        super().__init__()

        # self.norm_layer1 = nn.LayerNorm(in_features)

        self.gat_layer_1 = GATConvDense(
            num_in_features=in_features,
            num_out_features=hidden_features,
            num_of_heads=num_heads,
            concat=True,
            activation=nn.ReLU(),
            dropout_prob=dropout_prob
        )

        # self.activation1 = nn.ELU()

        # self.norm_layer2 = nn.LayerNorm(hidden_features * num_heads)

        self.gat_layer_2 = GATConvDense(
            num_in_features=hidden_features * num_heads,
            num_out_features=out_features,
            num_of_heads=1, # Thường lớp cuối cùng có 1 head hoặc avg
            concat=False,
            activation=None,
            dropout_prob=dropout_prob
        )

        if in_features != hidden_features * num_heads:
          self.skip1 = nn.Linear(in_features, hidden_features * num_heads)
        else:
          self.skip1 = nn.Identity()

        if hidden_features * num_heads != out_features:
          self.skip2 = nn.Linear(hidden_features * num_heads, out_features)
        else:
          self.skip1 = nn.Identity()

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, adj):
      # PreNorm Architecture
      # Norm -> GAT -> Dropout -> Residual -> Activation -> Norm -> GAT -> Residual 
        # x_skip1 = self.skip1(x)

        # h = self.norm_layer1(x)
        # h = self.gat_layer_1(h, adj)
        # h = self.dropout(h)
        # x = x_skip1 + h 

        # x = self.activation1(x)

        # x_skip2 = self.skip2(x)
        # h = self.norm_layer2(x)
        # h = self.gat_layer_2(h, adj)
        # x = x_skip2 + h   
             
        # return x

      # A simpler architecture
      h = self.gat_layer_1(x, adj)
      h = self.dropout(h)
      h = self.gat_layer_2(h, adj)

      return h 