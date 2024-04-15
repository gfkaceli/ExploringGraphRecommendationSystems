import csv
import os


def log_metrics_to_csv(file_path, metrics):
    """
    Log performance metrics to a CSV file.

    Parameters:
        file_path (str): The path to the CSV file where metrics will be logged.
        metrics (dict): A dictionary containing the metrics to log.
                        Expected keys: 'test_loss', 'mse', 'rmse', 'precision', 'recall'
    """
    # Check if the file exists already. If not, we need to write the header.
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode
    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())

        # Write the header if the file is new
        if not file_exists:
            writer.writeheader()

        # Write the metrics row
        writer.writerow(metrics)