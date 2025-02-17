"""Example with DeepDDs."""

import wandb

from chemicalx import pipeline
from chemicalx.data import LocalExampleDatasetLoader
from chemicalx.models import DeepDDS


def main():
    """Train and evaluate the DeepDDs model."""
    dataset = LocalExampleDatasetLoader('drugcombdb', 'dataset')
    wandb.init(project='ChemicalX',
               name='DeepDDS',
               tags=['baseline', 'example', 'DrugCombDB'])
    model = DeepDDS(
        context_channels=dataset.context_channels,
    )
    results = pipeline(
        dataset=dataset,
        model=model,
        batch_size=5120,
        epochs=10,
        context_features=True,
        drug_features=True,
        drug_molecules=True,
    )
    results.summarize()


if __name__ == "__main__":
    main()
