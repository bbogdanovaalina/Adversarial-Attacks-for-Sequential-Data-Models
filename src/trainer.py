import torch
from .tools import EarlyStopping, visual_data,  set_lr, cosine_annealing_lr, metrics
from .attacks import get_attack
from .datasets import get_dataset_loader
from .models import get_model
from tqdm.auto import tqdm


class Trainer:

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model = self._build_model(config).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adamax(self.model.parameters(), lr=config.lr)
        
    
    def _build_model(self, config):
        return get_model(config.model)(config)
    

    def _process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        preds = self.model(batch_x)

        loss = self.criterion(preds, batch_x)

        preds_class = preds.argmax(1)

        m = metrics(batch_y.cpu().detach(), preds_class.cpu().detach())

        return loss, m



    def train(self):
        self.model.train()
        train_set, train_loader = get_dataset_loader(self.config, 'TRAIN')
        val_set, val_loader = get_dataset_loader(self.config, 'TEST')
        early_stopping = EarlyStopping(self.config.patience, self.config.verbose, self.config.delta)
        train_loss_epoch = []
        val_loss_epoch = []


        for epoch in tqdm(range(self.config.num_epochs)):
            train_loss = []
            val_loss = []
            for batch_x, batch_y in train_loader:

                loss, m_train = self._process_one_batch(batch_x, batch_y)
                train_loss.append(loss.item())
                self.optim.zero_grad()

                loss.backward()

                self.optim.step()

            with torch.no_grad():
                for batch_x, batch_y in val_loader:

                    loss, m_val = self._process_one_batch(batch_x, batch_y)
                    val_loss.append(loss.item())

            early_stopping(-m_val['accuracy_score'], self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break



    def test(self, config):
        pass

    def visualize(self, config):
        pass

    def test_adverasial(self, config):

        pass