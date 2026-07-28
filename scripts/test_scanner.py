from src.datasets.scanner import DatasetScanner

scanner = DatasetScanner("datasets/raw")

samples = scanner.scan()

print(f"Found {len(samples)} patients")

for sample in samples[:5]:
    print(sample)