from src.datasets.scanner import DatasetScanner


def test_scanner_creation():
    scanner = DatasetScanner("datasets/raw")
    assert scanner is not None