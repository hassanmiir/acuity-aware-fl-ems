import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy


class FederatedClient:
    """
    Federated learning client representing a single ambulance.
    """

    def __init__(self, client_id, train_loader, test_loader,
                 model, config):
        self.client_id    = client_id
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.config       = config
        self.device       = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model        = copy.deepcopy(model).to(self.device)
        self.local_epochs = config["federated"]["local_epochs"]
        self.lr           = config["federated"]["learning_rate"]

        # BCEWithLogitsLoss expects RAW LOGITS — no sigmoid before loss
        # pos_weight corrects for 1.9% positive rate (~51x imbalance)
        pos_weight     = torch.tensor([51.0]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.staleness    = 0
        self.dataset_size = len(train_loader.dataset)
        self.acuity_score = 0.5

    def update_model(self, global_weights):
        self.model.load_state_dict(global_weights)
        self.staleness = 0

    def increment_staleness(self):
        self.staleness += 1

    def update_acuity(self, acuity_scorer):
        for batch_x, _ in self.train_loader:
            seq = batch_x.numpy()[:1]
            if len(seq) > 0:
                self.acuity_score = acuity_scorer.compute_score(seq[0])
            break

    def local_train(self):
        """
        Local training. Model outputs raw logits.
        BCEWithLogitsLoss applies sigmoid internally.
        """
        self.model.train()

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(self.local_epochs, 1),
            eta_min=self.lr * 0.1
        )

        for epoch in range(self.local_epochs):
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.float().to(self.device)

                optimizer.zero_grad()

                # model returns raw logits — DO NOT apply sigmoid here
                logits = self.model(batch_x)

                # Sanity check — catch NaN before backward
                if torch.isnan(logits).any():
                    continue

                loss = self.criterion(logits, batch_y)

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                optimizer.step()

            scheduler.step()

        return copy.deepcopy(self.model.state_dict()), \
               self.dataset_size

    def evaluate(self):
        """
        Evaluate local model. Apply sigmoid manually to logits
        since BCEWithLogitsLoss is used during training.
        """
        self.model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                logits  = self.model(batch_x)
                # Apply sigmoid here — NOT during training
                preds   = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())

        return np.array(all_preds), np.array(all_labels)