import time
from tqdm import tqdm
from modules import data_preprocessing as dataPrep

def benchmark_generator(generator, num_batches=None, name="Generator"):
    """
    Measures the data streaming and augmentation throughput of a sequence generator
    """
    if num_batches is None or num_batches > len(generator):
        num_batches = len(generator)

    print(f"--- Benchmarking {name} ---")
    print(f"{num_batches} batches (Batch size: {generator.batch_size})")

    start_time = time.time()

    # iterate through the generator
    for i in tqdm(range(num_batches), desc=f"Profiling {name}", unit="batch"):
        X, y = generator[i]
    
    end_time = time.time()

    total_time = end_time - start_time
    time_per_batch = total_time / num_batches
    batches_per_sec = num_batches / total_time
    samples_per_sec = (num_batches * generator.batch_size) / total_time

    print(f"\nTotal time: {total_time}")
    print(f"Time per batch: {time_per_batch}")
    print(f"Batches per second: {batches_per_sec}")
    print(f"Samples per second: {samples_per_sec}")

    return samples_per_sec

DATASET_DIR = '/home/marcel/Repos/Data/kits23_processed/kits_dataset_123_123_116_kidney'
fold_no = 0

train_paths, val_paths = dataPrep.get_train_validation_paths(DATASET_DIR, fold_no)


generator = dataPrep.KITSDataGenerator(train_paths, batch_size=4, augment=True)
benchmark_generator(generator, 10)


generator = dataPrep.KITSDataGenerator(train_paths, batch_size=4, augment=False)
benchmark_generator(generator, 10)