from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch.optim as optim
import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate, lr, wd, optimizer_name, class_weights_tensor):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

        # loss & optimizer
        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == "AdamW":
            self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == "RMSprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr, weight_decay=wd)

        # early stopping
        self.patience = 3
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop = False

        # losses
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        return self.net(x)

    def fit_epoch(self, train_loader, device):
        """ Trains the model for one epoch """

        self.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            self.optimizer.zero_grad()
            outputs = self.forward(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            self.train_losses.append(loss.item() / len(train_loader))

        return total_loss / len(train_loader)

    def check_early_stopping(self, val_loss):
        """ Returns True if training should stop """

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def validate(self, X_val, y_val, device):
        """ Calculates validation loss """

        self.eval()
        with torch.no_grad():
            outputs = self.forward(X_val.to(device))
            loss = self.criterion(outputs, y_val.to(device))

        self.val_losses.append(loss.item())
        return loss.item()
    
    def predict_and_evaluate(self, X_val, y_val, device):
        self.eval()
        with torch.no_grad():
            final_outputs = self.forward(X_val.to(device))
            probs = torch.softmax(final_outputs, dim=1)
            _, y_pred = torch.max(final_outputs, 1)

        # move back to CPU for metric calculation
        y_pred_cpu = y_pred.cpu().numpy()
        y_v_cpu = y_val.cpu().numpy()
        
        y_pred_cpu = y_pred.cpu().numpy()
        probs_cpu = probs.cpu().numpy()
        y_v_cpu = y_val.cpu().numpy()
        
        f1 = f1_score(y_v_cpu, y_pred_cpu, average='macro')
        acc = accuracy_score(y_v_cpu, y_pred_cpu)
        prec = precision_score(y_v_cpu, y_pred_cpu, average='macro', zero_division=0)
        rec = recall_score(y_v_cpu, y_pred_cpu, average='macro')
        roc_auc = roc_auc_score(y_v_cpu, probs_cpu, multi_class='ovr')
        
        return y_pred_cpu, probs_cpu, {
            'accuracy': acc, 
            'precision': prec, 
            'recall': rec, 
            'f1': f1, 
            'roc_auc': roc_auc
        }
