"""Example with DeepDDI."""

import wandb

from chemicalx import pipeline
from chemicalx.data import LocalExampleDatasetLoader
from chemicalx.models import DeepDDI


def main():
    """Train and evaluate the DeepSynergy model."""
    dataset = LocalExampleDatasetLoader('drugbankddi', 'dataset')
    wandb.init(project='ChemicalX',
               name='DeepDDI',
               tags=['baseline', 'example', 'DrugbankDDIs'])
    model = DeepDDI(drug_channels=dataset.drug_channels, hidden_layers_num=2)
    results = pipeline(
        dataset=dataset,
        model=model,
        batch_size=5120,
        epochs=100,
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
