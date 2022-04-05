from collections.abc import Sequence

from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score
from torch.optim import Adam

from chemicalx.data import dataset_resolver
from chemicalx.models import model_resolver
from chemicalx.pipeline import metric_resolver


class LightningPipeline(LightningModule):
    def __init__(self, dataset, model, model_kwargs, optimizer_cls=Adam, optimizer_kwargs=None,
                 loss_cls=None, loss_kwargs=None, batch_size=512, context_features=False,
                 drug_features=True, drug_molecules=False, train_size=None, random_state=None,
                 metrics=None):
        super(LightningPipeline, self).__init__()
        self.dataset = dataset
        self.model = model_resolver.make(model, model_kwargs)
        self.batch_size = batch_size
        self.context_features = context_features
        self.drug_features = drug_features
        self.drug_molecules = drug_molecules
        self.train_size = train_size
        self.random_state = random_state
        self.loss = loss_cls(**(loss_kwargs or {}))
        if metrics is None:
            self.metric_dict = {"roc_auc": roc_auc_score}
        else:
            self.metric_dict = {name: metric_resolver.lookup(name) for name in metrics}
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.model.parameters(), **(self.optimizer_kwargs or {}))
        return optimizer

    def training_step(self, batch, batch_idx):
        logists = self.model(*self.model.unpack(batch))
        loss_val = self.loss(logists, batch.labels)
        self.log('train/loss', loss_val, on_step=True, on_epoch=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        logists = self.model(*self.model.unpack(batch))
        loss_val = self.loss(logists, batch.labels)
        if isinstance(logists, Sequence):
            logists = logists[0]
        self.log('val/loss', loss_val, on_step=True, on_epoch=True)
        for key in self.metric_dict.keys():
            score = self.metric_dict[key](batch.labelss, logists)
            self.log('val/{}'.format(key), score, on_step=True, on_epoch=True)
        return loss_val

    def setup(self, stage):
        loader = dataset_resolver.make(self.dataset)
        self.train_generator, self.test_generator = loader.get_generators(
            batch_size=self.batch_size,
            context_features=self.context_features,
            drug_features=self.drug_features,
            drug_molecules=self.drug_molecules,
            train_size=self.train_size,
            random_state=self.random_state,
        )

    def train_dataloader(self):
        return self.train_generator

    def val_dataloader(self):
        return self.test_generator
