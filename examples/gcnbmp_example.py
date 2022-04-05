"""Example with GCNBMP."""

import wandb

from chemicalx import pipeline
from chemicalx.data import LocalExampleDatasetLoader
from chemicalx.models import GCNBMP


def main():
    """Train and evaluate the GCNBMP model."""
    dataset = LocalExampleDatasetLoader('drugcombdb', 'dataset')
    model = GCNBMP(hidden_conv_layers=2)
    wandb.init(project='ChemicalX',
               name='GCNBMP',
               tags=['baseline', 'example', 'DrugCombDB'])
    results = pipeline(
        dataset=dataset,
        model=model,
        optimizer_kwargs=dict(lr=0.01, weight_decay=10 ** -7),
        batch_size=5120,
        epochs=100,
        context_features=True,
        drug_features=True,
        drug_molecules=True,
    )
    results.summarize()


if __name__ == "__main__":
    main()
