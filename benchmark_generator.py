import time
import pandas as pd
from tqdm import tqdm
from modules import data_preprocessing as dataPrep
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from termcolor import colored

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

DATASET_DIR = '/home/marcel/Repos/Data/kits23_processed/kits_dataset_36_60_60_kidney'
fold_no = 0

train_paths, val_paths = dataPrep.get_train_validation_paths(DATASET_DIR, fold_no)

# generator = dataPrep.KITSDataGenerator(train_paths, batch_size=4, augment=True)
# benchmark_generator(generator, 10)

# generator = dataPrep.KITSDataGenerator(train_paths, batch_size=4, augment=False)
# benchmark_generator(generator, 10)

df = pd.DataFrame(columns=['batch_size', 'augment', 'samples_per_sec'])

for batch_size in range(1,17):
    for augment in [True, False]:
        print(f"batch size: {batch_size} | augment: {augment}")
        generator = dataPrep.KITSDataGenerator(train_paths, batch_size=batch_size, augment=augment, workers=4)
        sps = benchmark_generator(generator, 64)
        df.loc[len(df)] = [batch_size, augment, sps]

pivot_df = df.pivot(
    index='batch_size', columns='augment', values='samples_per_sec'
)
time_multiplier = pivot_df[False] / pivot_df[True]

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(pivot_df.index, time_multiplier, color='teal', s=50)

ax.set_title("Augmentation Time Multiplier by Batch Size", fontweight='bold', pad=10)
ax.set_xlabel("Batch Size")
ax.set_ylabel("Time Multiplier (Aug / No Aug)")

# use x-ticks to force integer batch sizes on x-axis
ax.set_xticks(pivot_df.index)

ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
out_dir = Path('./figures')
out_dir.mkdir(parents=True, exist_ok=True)
save_path = out_dir / 'augmentation_performance_impact.png'

plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(colored(f"[Success] Augmentation time multiplier plot saved to: {save_path}", "green"))