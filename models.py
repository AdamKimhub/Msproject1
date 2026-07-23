import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, NNConv, CGConv, global_max_pool, GCNConv, GATConv, GATv2Conv
import numpy as np
import matplotlib.pyplot as plt

# ====================
# THEMODELS
# ====================
NODE_DIM = 27
EDGE_DIM = 14
HIDDEN_DIM = 128
EMBED_DIM = 64
U_DIM = 14

# MODEL 0
class GNNModel0(nn.Module):
    def __init__(self, node_dim=8, edge_dim=3, hidden_dim=32, embed_dim=64):
        super().__init__()
        # Edge NN maps edge_attr to weight matrix
        self.edge_nn = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, node_dim * hidden_dim)
        )

        self.element_embedding = nn.Embedding(118, embed_dim)

        self.conv1 = NNConv(node_dim, hidden_dim, self.edge_nn, aggr='mean')

        self.fc0 = nn.Linear(hidden_dim + embed_dim, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def get_u(self, the_ids, the_ratios):
        emb = self.element_embedding(the_ids)
        weighted_emb = emb * the_ratios.unsqueeze(1)
        return weighted_emb.sum(dim=0)

    def forward(self,data):
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = global_mean_pool(x, data.batch)
        idss = data.the_ids.squeeze(0)
        ratioss = data.the_ratios.squeeze(0)
        global_attr = self.get_u(idss.to(self.element_embedding.weight.device),
                                 ratioss.to(self.element_embedding.weight.device)
                                 ).expand(x.size(0), -1)

        x = torch.cat([x, global_attr], dim=1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# MODEL 1
class GNNModel1(nn.Module):
    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM, u_dim=U_DIM):
        super().__init__()
        # For the edges
        
        # self.edge_nn = nn.Sequential(nn.Linear(edge_dim, 64),nn.ReLU(),nn.Linear(64, node_dim * hidden_dim))

        # self.conv0 = NNConv(node_dim, hidden_dim, self.edge_nn, aggr='mean')
        self.conv0 = GATv2Conv(node_dim, hidden_dim, edge_dim=edge_dim)
        
        self.conv1 = CGConv(hidden_dim, edge_dim, aggr='mean') # , batch_norm=True)
        self.conv2 = CGConv(hidden_dim, edge_dim, aggr='mean') # , batch_norm=True)
        self.conv3 = CGConv(hidden_dim, edge_dim, aggr='mean') # , batch_norm=True)

        self.global_embed = nn.Linear(u_dim, embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch, u = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            data.u,
        )

        x = F.relu(self.conv0(x, edge_index, edge_attr))
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))

        x = global_mean_pool(x, batch)

        u = self.global_embed(u)

        x = torch.cat([x, u], dim=1)

        the_return = self.fc(x)
        return the_return

# MODEL 2


# =====================
# LOSS FUNCTIONS
# =====================

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        ey_t = input-target
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))
    
class RelativeMSELoss(nn.Module):
    def __inti__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(((y_pred-y_true)/ (y_true + 1e-8)) ** 2)


class SMAPELoss(nn.Module):
    def __inti__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred))/2.0 + 1e-8
        return torch.mean(numerator/denominator)

class RelativeWeightedLoss(nn.Module):
    def __init__(self, delta=0.1, lambda_rel=0.3, eps=1e-2, alpha=1.5):
        super().__init__()
        self.delta = delta
        self.lambda_rel = lambda_rel  # weight of relative term
        self.eps = eps                # prevents div-by-zero for ~0 targets
        self.alpha = alpha

    def forward(self, input, target):
        error = input - target
        abs_error = torch.abs(error)

        # Huber absolute loss
        huber = torch.where(
            abs_error < self.delta,
            0.5 * error ** 2,
            self.delta * (abs_error - 0.5 * self.delta)
        )

        # Relative error: penalizes proportional misses
        relative = abs_error / (torch.abs(target) + self.eps)
        to_return = torch.mean(huber + self.lambda_rel * relative)

        # weights = (1.0 + target) ** self.alpha
        # weighted = weights * huber 
        # to_return = torch.mean(weighted)

        return to_return
    

# ====================================
# TRAINING AND VALIDATION FUNCTIONS
# ====================================

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, loader, loss_fn, device):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for data in loader:

            data = data.to(device)

            out = model(data)
            loss = loss_fn(out, data.y)

            eval_loss += loss.item() * data.num_graphs
    return eval_loss / len(loader.dataset)

def train(model, train_loader, loss_fn, device, optimizer):
    model.train()
    train_loss = 0

    for data in train_loader:
        optimizer.zero_grad()

        data = data.to(device)
        out = model(data)
        loss = loss_fn(out, data.y)

        loss.backward()

        optimizer.step()
        train_loss += loss.item() * data.num_graphs

    return train_loss / len(train_loader.dataset)

def predict(model, loader, device, visualize=False):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)

            if data.y:
                actuals.append(data.y.cpu().numpy())

            predictions.append(out.cpu().numpy())

    prediction = np.concatenate(predictions, axis=0).flatten()
    actual = np.concatenate(actuals, axis=0).flatten()

    if visualize:
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # model values
        predicted_values = prediction
        actual_values = actual
        residuals = actual_values - predicted_values


        axs[0].scatter(actual_values, predicted_values )
        axs[0].plot(actual_values, actual_values, c="black")
        axs[0].set_xlabel('Actual Band Gaps(eV)')
        axs[0].set_ylabel('Predicted Band Gaps(eV)')
        axs[0].set_title('Actual vs Predicted Band Gap Values')
        axs[0].legend()

        axs[1].plot(actual_values[100:150], label='Actual Values', marker='o')
        axs[1].plot(predicted_values[100:150], label='Predicted Values', marker='o')
        axs[1].set_xlabel('Samples')
        axs[1].set_ylabel('Band gap(eV)')
        axs[1].set_title('Actual vs Predicted Band Gap Values')
        axs[1].legend()

        axs[2].scatter(predicted_values, residuals )
        axs[2].axhline(0, color='red', linestyle='--')
        axs[2].set_xlabel('Predicted Band Gaps(eV)')
        axs[2].set_ylabel('Residuals (Actual - Predicted)')
        axs[2].set_title('Residual Plot')

        plt.tight_layout()
        plt.show()

    return prediction, actual

# ==============================
# COMMON PARAMETERS
# ==============================

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model0 = GNNModel0().to(device)
# model1 = GNNModel1().to(device)
# optimizer = torch.optim.Adam(.parameters(), lr=0.001)

# Loss function
# loss_fn = nn.MSELoss()