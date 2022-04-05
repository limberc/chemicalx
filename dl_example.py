from tqdm import tqdm

from chemicalx.data import LocalExampleDatasetLoader, dataset_resolver

if __name__ == '__main__':
    dataset_names = ["drugcombdb", "drugcomb", "twosides", "drugbankddi"]
    print("Testing dataset.")
    for dataset_name in tqdm(dataset_names):
        dataset = LocalExampleDatasetLoader(dataset_name, 'dataset')
        loader = dataset_resolver.make(dataset)
        train_generator, test_generator = loader.get_generators(
            batch_size=32,
            context_features=True,
            drug_features=True,
            drug_molecules=True,
            train_size=0.2,
        )
