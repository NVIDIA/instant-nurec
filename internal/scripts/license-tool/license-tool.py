import argparse
import datetime
import os
import re
import subprocess

from enum import Enum


LICENSE_TEMPLATE = """SPDX-FileCopyrightText: Copyright (c) YEAR_STRING NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited."""

# Accepted headers that should not trigger interactive prompts
ACCEPTED_HEADERS = [
    # Full NVIDIA header with "Copyright YEAR_STRING," format
    """Copyright YEAR_STRING, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-FileCopyrightText: Copyright (c) YEAR_STRING NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.""",
    # Short NVIDIA copyright header
    """Copyright (c) YEAR_STRING NVIDIA CORPORATION.  All rights reserved.""",
]

COMMENT_CONFIG = {
    ".c": {"line": "//", "block_start": "/*", "block_end": "*/"},
    ".cpp": {"line": "//", "block_start": "/*", "block_end": "*/"},
    ".cc": {"line": "//", "block_start": "/*", "block_end": "*/"},
    ".cxx": {"line": "//", "block_start": "/*", "block_end": "*/"},
    ".h": {"line": "//", "block_start": "/*", "block_end": "*/"},
    ".hpp": {"line": "//", "block_start": "/*", "block_end": "*/"},
    ".cu": {"line": "//", "block_start": "/*", "block_end": "*/"},
    ".cuh": {"line": "//", "block_start": "/*", "block_end": "*/"},
    ".py": {"line": "#"},
}


class HeaderStatus(Enum):
    MISSING = 1
    DIFFERENT = 2
    ASEXPECTED = 3


def normalize_header_for_comparison(header, comment_style):
    """Normalize a header by removing comment markers and normalizing whitespace."""
    if not header:
        return ""

    lines = header.split("\n")
    normalized_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped:
            # Remove comment markers
            if "line" in comment_style and stripped.startswith(comment_style["line"]):
                stripped = stripped[len(comment_style["line"]) :].strip()
            elif "block_start" in comment_style and stripped.startswith(comment_style["block_start"]):
                stripped = stripped[len(comment_style["block_start"]) :].strip()
            elif "block_end" in comment_style and stripped.endswith(comment_style["block_end"]):
                stripped = stripped[: -len(comment_style["block_end"])].strip()

            if stripped:  # Only add non-empty lines
                normalized_lines.append(stripped)

    return "\n".join(normalized_lines)


def is_header_accepted(header, comment_style):
    """Check if the given header matches any of the accepted headers."""
    if not header:
        return False

    normalized_header = normalize_header_for_comparison(header, comment_style)

    for accepted_template in ACCEPTED_HEADERS:
        # Normalize the accepted template first
        accepted_normalized = normalize_header_for_comparison(accepted_template, comment_style)

        # For simple string matching, replace placeholders with wildcards
        # Replace YEAR_STRING with a pattern that matches any year
        pattern = accepted_normalized.replace("YEAR_STRING", "YEAR_PLACEHOLDER")
        pattern = pattern.replace("[author name]", "AUTHOR_PLACEHOLDER")

        # Check if the structure matches by comparing non-placeholder parts
        # Split both strings and compare non-placeholder parts
        header_parts = normalized_header.split()
        pattern_parts = pattern.split()

        if len(header_parts) == len(pattern_parts):
            match = True
            for i, (header_part, pattern_part) in enumerate(zip(header_parts, pattern_parts)):
                if pattern_part == "YEAR_PLACEHOLDER" or pattern_part.startswith("YEAR_PLACEHOLDER"):
                    # Check if this part looks like a year (with optional trailing punctuation)
                    if not re.match(r"\d{4}(-\d{4})?[,;.]?$", header_part):
                        match = False
                        break
                    # Also check if the punctuation matches
                    pattern_punct = pattern_part.replace("YEAR_PLACEHOLDER", "")
                    header_punct = re.sub(r"\d{4}(-\d{4})?", "", header_part)
                    if pattern_punct != header_punct:
                        match = False
                        break
                elif pattern_part == "AUTHOR_PLACEHOLDER":
                    # Any non-empty string is acceptable for author
                    if not header_part:
                        match = False
                        break
                elif header_part != pattern_part:
                    match = False
                    break

            if match:
                return True

    return False


def is_difference_only_year(expected_header, actual_header, comment_style):
    """Check if the only difference between headers is in the year string."""
    if not actual_header:
        return False

    # Normalize both headers
    expected_normalized = normalize_header_for_comparison(expected_header, comment_style)
    actual_normalized = normalize_header_for_comparison(actual_header, comment_style)

    # Replace year patterns with a placeholder
    year_pattern = r"\d{4}(-\d{4})?"
    expected_no_year = re.sub(year_pattern, "YEAR_PLACEHOLDER", expected_normalized)
    actual_no_year = re.sub(year_pattern, "YEAR_PLACEHOLDER", actual_normalized)

    # Compare the headers without year information
    return expected_no_year.strip() == actual_no_year.strip()


def list_files_with_extensions(directory, extensions):
    extensions_set = set(extensions)
    matching_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in extensions_set:
                matching_files.append(os.path.join(root, file))

    return matching_files


def extract_git_years(file_path):
    try:
        # Run the git command using subprocess
        absolute_parent_directory = os.path.dirname(os.path.abspath(file_path))
        cmd = "git log -C --follow --format=%ad --date default"
        cmd = "git -C " + absolute_parent_directory + " log --follow --format=%ad --date default"
        result = subprocess.run(cmd.split() + [file_path], capture_output=True, text=True, check=True)

        # Get the output from the command
        output = result.stdout.strip()

        # Split the output into lines
        dates = output.split("\n")

        if not dates or not dates[0]:
            return None, None
        try:
            # Extract the year from the last edit (first date in the list)
            last_edit_year = dates[0].split(" ")[-2]

            # Extract the year from the creation date (last date in the list)
            creation_year = dates[-1].split(" ")[-2]

            return creation_year.strip(), last_edit_year.strip()
        except IndexError:
            return None, None
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None, None


def get_file_header(file_path, comment_style):
    header = ""
    shebang = None
    with open(file_path, "r") as file:
        lines = file.readlines()

    if not lines:
        return "", None

    start_line_idx = 0
    if lines[0].startswith("#!"):
        shebang = lines[0]
        start_line_idx = 1

    content_lines = lines[start_line_idx:]
    if not content_lines:
        return "", shebang

    # Find the first non-empty line
    first_content_idx = -1
    for i, line in enumerate(content_lines):
        if line.strip():
            first_content_idx = i
            break

    # If all lines are empty or whitespace
    if first_content_idx == -1:
        return "", shebang

    # Start processing from the first line with content
    content_lines_from_first_content = content_lines[first_content_idx:]
    first_line_stripped = content_lines_from_first_content[0].strip()

    # Block comment check
    if "block_start" in comment_style and first_line_stripped.startswith(comment_style["block_start"]):
        for line in content_lines_from_first_content:
            header += line
            if comment_style["block_end"] in line:
                break
        return header, shebang

    # Line comment check
    if "line" in comment_style:
        # Check if the first content line is a comment
        if not first_line_stripped.startswith(comment_style["line"]):
            return "", shebang  # No header if first line is not a comment

        header = ""
        for line in content_lines_from_first_content:
            stripped = line.strip()

            if stripped.startswith("#!"):
                break

            # A line is part of the header if it's a comment.
            # A blank line is not a comment and terminates the header.
            if stripped.startswith(comment_style["line"]):
                header += line
            else:
                break
        return header, shebang

    return "", shebang


def get_header_status(file_path, expected_header, comment_style):
    # Split the header string into individual lines to determine the number of lines to read
    header_lines_to_compare = expected_header.split("\n")
    num_header_lines = len(header_lines_to_compare)

    header, shebang = get_file_header(file_path, comment_style)

    # Compare the header from the file with the given multi-line header string
    header_status = HeaderStatus.DIFFERENT
    if header == "":
        header_status = HeaderStatus.MISSING
    elif header.strip() == expected_header.strip():
        header_status = HeaderStatus.ASEXPECTED
    return header_status, header, shebang


def is_file_in_any_subdirectory(file, directory, subdirectories):
    # Normalize the file path
    file_path = os.path.abspath(file)

    # Iterate through each subdirectory in the list
    for subdirectory in subdirectories:
        # Construct the full path to the subdirectory
        subdirectory_path = os.path.join(directory, subdirectory)

        # Normalize the subdirectory path
        subdirectory_path = os.path.abspath(subdirectory_path)

        # Check if the file is in the current subdirectory or its subdirectories
        if os.path.commonpath([subdirectory_path, file_path]) == subdirectory_path:
            return True

    # If the file is not found in any of the subdirectories, return False
    return False


def modify_header(file_path, new_header, old_header, shebang):
    n = 0
    if old_header.strip() != "" and old_header.strip() != "\n":
        n = len(old_header.splitlines())

    # Read the contents of the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    start_modify_idx = 1 if shebang else 0
    new_lines = lines[:start_modify_idx]
    content_lines = lines[start_modify_idx:]

    # Remove the first n lines
    del content_lines[:n]

    # Get rid of extra white space lines:
    start_index = 0
    for i, line in enumerate(content_lines):
        if line.strip():  # Check if the line contains any non-whitespace characters
            start_index = i
            break
    del content_lines[:start_index]

    # Insert the string at the beginning of the list
    content_lines.insert(0, new_header + "\n\n")

    new_lines.extend(content_lines)

    # Write the modified content back to the file
    with open(file_path, "w") as file:
        file.writelines(new_lines)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="List files with specific extensions in a directory recursively.")

    # Add the arguments
    parser.add_argument("directory", type=str, help="The directory to search in")
    parser.add_argument("extensions", type=str, help="Comma-separated list of file extensions (e.g., .txt,.jpg,.png)")
    parser.add_argument(
        "--subdirs_to_skip",
        type=str,
        default="",
        help="Comma-separated list of subdirectories to ignore, full relative path (e.g. subdir1,subdir/subdir/subdir2",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        help="Replacement mode. 'silent' will replace any header that is missing or not as expected. 'interactive' will add all missing headers, but will ask the user for confirmation of replacement when they differ.",
    )
    parser.add_argument(
        "--write",
        type=bool,
        default=False,
        help="Whether or not to write out the modified files. If false, the app will report which files would have been modified",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Split the extensions by comma and strip spaces
    extensions_list = [ext.strip() for ext in args.extensions.split(",")]

    # Get the matching files
    result_files = list_files_with_extensions(args.directory, extensions_list)

    skip_subdirs = list(filter(None, args.subdirs_to_skip.split(",")))

    if args.mode not in ("silent", "interactive"):
        parser.error(f"Invalid mode: {args.mode}. Must be 'silent' or 'interactive'")

    # Print the matching files
    files_status = {}
    files_status["added"] = []
    files_status["replaced"] = []
    files_status["skipped"] = []
    files_status["skipped-empty"] = []
    files_status["marked"] = []
    files_status["no-action"] = []
    files_status["inserted"] = []
    for file in result_files:
        if len(skip_subdirs) != 0 and is_file_in_any_subdirectory(file, args.directory, skip_subdirs):
            continue

        if os.path.getsize(file) == 0:
            files_status["skipped-empty"].append(file)
            continue

        ext = os.path.splitext(file)[1]
        if ext not in COMMENT_CONFIG:
            print(f"Skipping file {file} as it has an unsupported extension.")
            continue

        comment_style = COMMENT_CONFIG[ext]

        year_1, year_2 = extract_git_years(file)
        if year_1 is None or year_2 is None:
            now = datetime.datetime.now()
            year_1 = now.year
            year_2 = now.year

        year_string = str(year_1) + "-" + str(year_2)
        if year_1 == year_2:
            year_string = str(year_1)

        license_text_with_year = LICENSE_TEMPLATE.replace("YEAR_STRING", str(year_string))

        lines = license_text_with_year.split("\n")
        comment_marker = comment_style["line"]

        commented_lines = []
        for line in lines:
            if line.strip() == "":
                commented_lines.append(comment_marker)
            else:
                commented_lines.append(f"{comment_marker} {line}")

        file_header_expected = "\n".join(commented_lines)

        # Get the header status (is it as expected, different, or missing
        header_status, header, shebang = get_header_status(file, file_header_expected, comment_style)
        file_write = False

        # Check if we should skip interactive prompt
        skip_interactive = False
        if header_status == HeaderStatus.DIFFERENT and args.mode == "interactive":
            # Auto-replace if header is in accepted list
            if is_header_accepted(header, comment_style):
                print(f"Auto-replacing {file} - header is in accepted list")
                files_status["replaced"].append(file)
                file_write = True
                skip_interactive = True
            # Auto-replace if difference is only in year
            elif is_difference_only_year(file_header_expected, header, comment_style):
                print(f"Auto-replacing {file} - difference is only in year")
                files_status["replaced"].append(file)
                file_write = True
                skip_interactive = True

        if header_status == HeaderStatus.DIFFERENT and args.mode == "interactive" and not skip_interactive:
            print("----------------------------------------------------------")
            print("Desired header and current header differ:\nDesired Header:")
            print(file)
            print(file_header_expected)
            print("Current header:\n" + header)
            user_input = input(
                "Do you wish to replace the header (press 'r'), skip (press 's'), insert above existing (press 'i'), or skip and mark (press 'm') ?"
            )
            if user_input == "r":
                files_status["replaced"].append(file)
                file_write = True
            elif user_input == "m":
                files_status["marked"].append(file)
            elif user_input == "i":
                files_status["inserted"].append(file)
                header = ""  # reset so that it doesn't get removed in "modify_header" call
            else:
                files_status["skipped"].append(file)
            print("----------------------------------------------------------")
        elif header_status == HeaderStatus.DIFFERENT and args.mode == "silent":
            files_status["replaced"].append(file)
            file_write = True
        elif header_status == HeaderStatus.MISSING:
            files_status["added"].append(file)
            file_write = True
        elif not skip_interactive:
            files_status["no-action"].append(file)

        if args.write and file_write:
            modify_header(file, file_header_expected, header, shebang)

    print(files_status)


if __name__ == "__main__":
    main()
