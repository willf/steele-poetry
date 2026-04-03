# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "datasets",
#     "huggingface-hub",
#     "Pillow",
# ]
# ///

import os
import re
import argparse
from datasets import Dataset, Image

def extract_page_number(filename):
    """
    Extracts the page number from the filename.
    Assumes the number is right before the .jp2 extension.
    """
    match = re.search(r'(\d+)\.jp2$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.findall(r'\d+', filename)
    if match:
        return int(match[-1])

    return 0

def main():
    parser = argparse.ArgumentParser(description="Create and upload an image dataset to Hugging Face using uv.")
    parser.add_argument("-d", "--dir", required=True, help="Root directory containing the subdirectories of images.")
    parser.add_argument("-r", "--repo", required=True, help="Hugging Face repository ID (e.g., 'your-username/my-steel-dataset').")
    parser.add_argument("--private", action="store_true", help="Set this flag to make the dataset private on Hugging Face.")

    args = parser.parse_args()

    image_paths = []
    page_names = []
    page_numbers = []
    directory_names = []

    print(f"Scanning directory: {args.dir}...")

    for root, dirs, files in os.walk(args.dir):
        for file in sorted(files):
            if file.lower().endswith('.jp2'):
                filepath = os.path.join(root, file)
                dir_name = os.path.basename(root)
                page_num = extract_page_number(file)

                image_paths.append(filepath)
                page_names.append(file)
                page_numbers.append(page_num)
                directory_names.append(dir_name)

    if not image_paths:
        print("No .jp2 files found. Please check your directory path.")
        return

    print(f"Found {len(image_paths)} images. Building dataset...")

    data_dict = {
        "image": image_paths,
        "page_name": page_names,
        "page_number": page_numbers,
        "directory_name": directory_names
    }

    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.cast_column("image", Image())

    print(f"Uploading dataset to Hugging Face Hub as '{args.repo}'...")

    dataset.push_to_hub(
        args.repo,
        private=args.private
    )

    print("Upload complete!")

if __name__ == "__main__":
    main()
