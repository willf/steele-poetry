# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
# ]
# ///

import argparse
from datasets import load_dataset


def int_to_roman(num):
    """Converts an integer to a lowercase Roman numeral."""
    values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    symbols = ["m", "cm", "d", "cd", "c", "xc", "l", "xl", "x", "ix", "v", "iv", "i"]
    roman_num = ""
    i = 0
    while num > 0:
        for _ in range(num // values[i]):
            roman_num += symbols[i]
            num -= values[i]
        i += 1
    return roman_num


def parse_ranges(range_args):
    """
    Parses range arguments like '4-17:roman' into a dictionary mapping
    the file index to its formatted page string.
    """
    page_mapping = {}

    if not range_args:
        return page_mapping

    for r in range_args:
        try:
            range_part, num_type = r.split(":")
            start_str, end_str = range_part.split("-")
            start = int(start_str)
            end = int(end_str)
            num_type = num_type.lower()

            page_counter = 1
            for i in range(start, end + 1):
                if num_type == "roman":
                    page_str = int_to_roman(page_counter)
                elif num_type == "arabic":
                    page_str = str(page_counter)
                else:
                    raise ValueError(
                        f"Unknown numbering type: {num_type}. Use 'roman' or 'arabic'."
                    )

                page_mapping[i] = page_str
                page_counter += 1

        except ValueError as e:
            print(
                f"Error parsing range '{r}'. Ensure format is start-end:type (e.g., 4-17:roman). Details: {e}"
            )
            exit(1)

    return page_mapping


def main():
    parser = argparse.ArgumentParser(
        description="Extract and combine markdown from a Hugging Face dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Hugging Face dataset name (e.g., 'username/dataset').",
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split to use (default: train)."
    )
    parser.add_argument(
        "-d", "--dir_name", required=True, help="The directory_name to filter by."
    )
    parser.add_argument(
        "-o", "--out", required=True, help="Output file path (e.g., combined.md)."
    )
    parser.add_argument(
        "-r",
        "--range",
        nargs="+",
        help="Page ranges and types. Format: start-end:type",
        default=[],
    )

    args = parser.parse_args()
    page_mapping = parse_ranges(args.range)

    print(f"Loading dataset '{args.dataset}' (split: {args.split})...")
    try:
        ds = load_dataset(args.dataset, split=args.split)

        # OPTIMIZATION: Drop heavy image/audio columns before filtering.
        # This prevents PIL.Image decoding overhead during the row iteration.
        columns_to_keep = ["directory_name", "page_number", "markdown"]
        actual_columns = ds.column_names
        keep = [c for c in columns_to_keep if c in actual_columns]

        print(f"Optimizing memory... isolating columns: {keep}")
        ds = ds.select_columns(keep)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Filtering for directory_name == '{args.dir_name}'...")
    filtered_ds = ds.filter(lambda x: x.get("directory_name") == args.dir_name)

    if len(filtered_ds) == 0:
        print(f"No pages found for directory_name '{args.dir_name}'.")
        return

    # Ensure the pages are in the correct order
    print("Sorting by page_number...")
    sorted_ds = filtered_ds.sort("page_number")

    print(f"Found {len(sorted_ds)} pages. Combining into {args.out}...")

    with open(args.out, "w", encoding="utf-8") as outfile:
        for row in sorted_ds:
            img_num_int = row.get("page_number")
            content = row.get("markdown", "")

            if img_num_int is None:
                print("Warning: Missing 'page_number' for a row. Skipping.")
                continue

            # Pad the image number with zeros for the header (e.g., 4 -> "0004")
            img_num_str = f"{img_num_int:04d}"

            page_str = page_mapping.get(img_num_int)

            if page_str:
                header = f"[IMAGE {img_num_str}; PAGE {page_str}]\n\n"
            else:
                header = f"[IMAGE {img_num_str}]\n\n"

            outfile.write(header)
            outfile.write(content)

            # Ensure there's a visual separation between pages
            if not content.endswith("\n"):
                outfile.write("\n")
            outfile.write("\n---\n\n")

    print("Done!")


if __name__ == "__main__":
    main()
