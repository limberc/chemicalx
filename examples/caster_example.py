"""Example with CASTER."""

import wandb

from chemicalx import pipeline
from chemicalx.data import LocalExampleDatasetLoader
from chemicalx.loss import CASTERSupervisedLoss
from chemicalx.models import CASTER


def main():
    """Train and evaluate the CASTER model."""
    wandb.init(project='ChemicalX',
               name='CASTER',
               tags=['baseline', 'example', 'DrugCombDB'])
    # dataset = DrugCombDB()
    dataset = LocalExampleDatasetLoader('drugcombdb', 'dataset')
    model = CASTER(drug_channels=dataset.drug_channels)
    results = pipeline(
        dataset=dataset,
        model=model,
        loss_cls=CASTERSupervisedLoss,
        batch_size=5120,
        epochs=1,
        context_features=False,
        drug_features=True,
        drug_molecules=False,
        metrics=[
            "roc_auc",
        ],
    )
    results.summarize()


if __name__ == "__main__":
    main()
