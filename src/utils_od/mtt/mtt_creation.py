import os
import zipfile
import shutil
import math
import csv

import requests
from tqdm import tqdm
from pydub import AudioSegment


def download_mtt(in_path_dl: str):
    """
    Download the MagnaTagATune (MTT) dataset's annotation file and mp3 data files.

    The dataset is split into multiple parts. This function will download all the parts
    and the annotation file.

    Args:
        in_path_dl (str):
            Directory path where the downloaded files will be saved.

    """
    # The URL of the file to download
    url_tags = "https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv"
    url_data_1 = "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.001"
    url_data_2 = "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.002"
    url_data_3 = "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.003"
    _download_file(url_tags, os.path.join(in_path_dl, 'annotations_final.csv'))
    _download_file(url_data_1, os.path.join(in_path_dl, 'mp3.zip.001'))
    _download_file(url_data_2, os.path.join(in_path_dl, 'mp3.zip.002'))
    _download_file(url_data_3, os.path.join(in_path_dl, 'mp3.zip.003'))


def _download_file(in_url: str, in_path_dl: str, chunk_size: int = 10485760, timeout: int = 5):
    """
    Utility function to download a file from a given URL and save it to a specified path.

    Args:
        in_url (str):
            URL of the file to be downloaded.
        in_path_dl (str):
            File path where the downloaded file will be saved.
        chunk_size (int, optional):
            Size of chunks to use while downloading. Default is 10 MB.
        timeout (int, optional):
            Timeout in seconds for the request. Default is 5 seconds.

    """
    try:
        # Streaming GET request
        response = requests.get(in_url, stream=True, timeout=timeout)
        response.raise_for_status()

        # Get the total file size from headers
        total_size = int(response.headers.get('content-length', 0))

        # Create a progress bar
        pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=in_path_dl)

        # Download the file in chunks and write to the file sequentially
        with open(in_path_dl, "wb") as file:
            for data in response.iter_content(chunk_size=chunk_size):
                file.write(data)
                pbar.update(len(data))
        pbar.close()

    except requests.exceptions.Timeout:
        print(f"The request timed out for {in_url}.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


def unzip_mtt(in_dir: str):
    """
    Unzip the concatenated MagnaTagATune (MTT) dataset's mp3 data files.

    The dataset is split into multiple parts. This function expects these parts to
    be already downloaded and present in the specified directory. It first concatenates
    these parts into a single zip file and then extracts the contents.

    Args:
        in_dir (str):
            Directory path where the downloaded parts are present and where the extracted
            files will be saved in a 'raw' subdirectory.

    """
    # Define the parts and the concatenated zip file path
    parts = [
        os.path.join(in_dir, 'mp3.zip.001'),
        os.path.join(in_dir, 'mp3.zip.002'),
        os.path.join(in_dir, 'mp3.zip.003')
    ]
    concatenated_zip = os.path.join(in_dir, 'mp3_combined.zip')
    # Concatenate the parts into one zip file
    with open(concatenated_zip, 'wb') as output_file:
        for part in parts:
            with open(part, 'rb') as input_file:
                output_file.write(input_file.read())

    path_unzip_dest = os.path.join(in_dir, 'raw')
    os.makedirs(path_unzip_dest, exist_ok=True)

    # Unzip the concatenated file
    with zipfile.ZipFile(concatenated_zip, 'r') as zip_ref:
        # Using tqdm to display the progress
        for member in tqdm(zip_ref.infolist(), desc="Unzipping", unit="file"):
            zip_ref.extract(member, path_unzip_dest)  # Use path_unzip_dest here


def get_histogram(in_csv_file_path: str):
    """
     Parse the given CSV file, calculate the frequency of each tag (excluding 'clip_id'),
     and save the histogram of tag counts in descending order to "tag_histogram.txt"
     in the same directory as the input CSV.

     The function expects the CSV file to contain multiple columns where each column
     represents a tag (excluding 'clip_id'). The values in each column should be integers
     indicating the number of occurrences of that tag.

     Args:
         in_csv_file_path (str):
             Path to the CSV file containing tags and their corresponding counts.

     """
    # Dictionary to store tag counts
    tag_counts = {}

    # Read the CSV file
    with open(in_csv_file_path, 'r', encoding="utf-8") as csvfile:
        reader: csv.DictReader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            for tag, value in row.items():
                # skip the clip_id column
                if tag != 'clip_id':
                    try:
                        # Attempt to convert value to integer
                        value = int(value)
                        if tag in tag_counts:
                            tag_counts[tag] += value
                        else:
                            tag_counts[tag] = value
                    except ValueError:
                        # If value is not an integer, skip processing it
                        continue

    # Sort the tags by their counts in descending order
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

    path_save = os.path.join(os.path.dirname(in_csv_file_path), "tag_histogram.txt")
    # Save tag counts to tag_histogram.txt
    with open(path_save, 'w', encoding="utf-8") as f:
        for tag, count in sorted_tags:
            f.write(f"{tag}: {count}\n")

    print(f"Tag histogram saved to {path_save}.")


def copy_exclusive_tags(in_csv_file_path, in_classes_mapping, in_dir_source, in_dir_dest):
    """
    Parse the given CSV file, identify the exclusive tags for each class, and
    copy the associated files from the source directory to the destination directory,
    maintaining a train/test split.

    The function expects the CSV file to contain multiple columns. The last column
    should contain file paths, and the other columns should be matched against the
    provided class synonyms. Files are copied to the destination under "train" and
    "test" subdirectories, split approximately 90/10.

    Directory structure of the destination:
        root
        - train
            - class_1
            - class_2
            ...
        - test
            - class_1
            - class_2
            ...

    Args:
        in_csv_file_path (str):
            Path to the CSV file containing file paths and class synonyms.
        in_classes_mapping (dict):
            Dictionary mapping class names to lists of synonyms present in the CSV file.
        in_dir_source (str):
            Source directory containing the files to be copied.
        in_dir_dest (str):
            Destination directory to which the files will be copied, maintaining
            a class-based structure with train/test split.

    """
    # Dictionary to store paths for each class
    exclusive_paths = {cls: [] for cls in in_classes_mapping.keys()}

    # Read the CSV file
    with open(in_csv_file_path, 'r', encoding="utf-8") as csvfile:
        reader: csv.DictReader = csv.DictReader(csvfile, delimiter='\t')

        for row in reader:
            # This will store which classes had any tags matched
            matched_classes = []

            for cls, synonyms in in_classes_mapping.items():
                # If any synonym has a tag for this row, note the class
                if any(int(row[synonym]) for synonym in synonyms if synonym in row and row[synonym].isdigit()):
                    matched_classes.append(cls)

            # If only one class is matched, it's an exclusive tag
            if len(matched_classes) == 1:
                # Assuming the last column contains the path
                path = row[reader.fieldnames[-1]]
                exclusive_paths[matched_classes[0]].append(path)

    # Create necessary directories first
    for cls in in_classes_mapping.keys():
        os.makedirs(os.path.join(in_dir_dest, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(in_dir_dest, 'test', cls), exist_ok=True)

    total_files = sum(len(paths) for paths in exclusive_paths.values())
    with tqdm(total=total_files, desc="Copying", unit="file") as pbar:
        for cls, paths in exclusive_paths.items():
            # Calculate the index where to split
            split_index = math.ceil(0.9 * len(paths))

            train_paths = paths[:split_index]
            test_paths = paths[split_index:]

            # Copy train files
            for path in train_paths:
                src_path = os.path.join(in_dir_source, path)
                dst_path = os.path.join(in_dir_dest, 'train', cls, os.path.basename(path))
                shutil.copy2(src_path, dst_path)
                pbar.update(1)  # Update progress bar

            # Copy test files
            for path in test_paths:
                src_path = os.path.join(in_dir_source, path)
                dst_path = os.path.join(in_dir_dest, 'test', cls, os.path.basename(path))
                shutil.copy2(src_path, dst_path)
                pbar.update(1)  # Update progress bar

    print("Files copied successfully!")


def convert_mp3_to_wav(in_source_root, in_target_root, overwrite=True):
    # Pre-create all directories
    for dirpath, _, _ in os.walk(in_source_root):
        relative_path = os.path.relpath(dirpath, in_source_root)
        target_dirpath = os.path.join(in_target_root, relative_path)
        os.makedirs(target_dirpath, exist_ok=True)

    # Get the list of all mp3 files for progress bar
    mp3_files = [os.path.join(dirpath, filename)
                 for dirpath, _, filenames in os.walk(in_source_root)
                 for filename in filenames if filename.endswith('.mp3')]

    # Convert .mp3 to .wav with progress bar
    for source_filepath in tqdm(mp3_files, desc="Converting", unit="file"):
        # Construct the target filepath
        dirpath = os.path.dirname(source_filepath)
        filename = os.path.basename(source_filepath)

        relative_path = os.path.relpath(dirpath, in_source_root)
        target_dirpath = os.path.join(in_target_root, relative_path)
        target_filename = filename.replace('.mp3', '.wav')
        target_filepath = os.path.join(target_dirpath, target_filename)

        # Check for overwrite
        if not overwrite and os.path.exists(target_filepath):
            continue

        # Convert the file with error handling
        try:
            audio = AudioSegment.from_mp3(source_filepath)
            audio.export(target_filepath, format="wav")
        # We are not interested in handling specific exceptions, if any happens just skip the file
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error processing {source_filepath}: {e}")


# Download and unzip official MagnaTagATune dataset
PATH_DOWNLOAD = "../../../data/mtt/"
download_mtt(PATH_DOWNLOAD)
unzip_mtt(PATH_DOWNLOAD)

# Print out histogram of tags, to see which super-classes make sense
PATH_CSV = "../../../data/mtt/annotations_final.csv"
get_histogram(PATH_CSV)

# Construct MTT subset consisting of 5 classes, only tracks that have no class-overlapping tags are selected
CLASSES_MAPPING = {
    "classical": ["classical", "clasical", "classic"],
    "techno": ["techno", "synth", "electro", "electric"],
    "rock": ["rock", "hard rock"],
    "indian": ["indian"],
    "pop": ["pop"]
}
copy_exclusive_tags(PATH_CSV, CLASSES_MAPPING, in_dir_source=os.path.join(PATH_DOWNLOAD, 'raw'), in_dir_dest=os.path.join(PATH_DOWNLOAD, 'od_subset'))
convert_mp3_to_wav(in_source_root=os.path.join(PATH_DOWNLOAD, 'od_subset'), in_target_root=os.path.join(PATH_DOWNLOAD, 'mtt_wav'))
