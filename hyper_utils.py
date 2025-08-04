import re
import argparse

def filter_trials(input_file, val_thresh=3e-4, gap_thresh=9e-5):
    output_file = "filtered_trials.txt"

    # Regex patterns
    trial_line_pattern = re.compile(
        r"\[Trial (\d+)\] Train loss: ([\deE\.-]+), Val loss: ([\deE\.-]+), Overfit gap: ([\deE\.-]+)"
    )
    optuna_line_pattern = re.compile(
        r"\[I .*?Trial (\d+)\s+finished with value: .*?parameters: .*"
    )

    # Store Optuna lines by trial number
    optuna_lines = {}
    output_lines = []

    with open(input_file, "r") as f:
        lines = [line.strip() for line in f]

    # First, collect all Optuna lines
    for line in lines:
        match = optuna_line_pattern.match(line)
        if match:
            trial_num = int(match.group(1))
            optuna_lines[trial_num] = line

    # Then process trial lines and filter
    for line in lines:
        match = trial_line_pattern.match(line)
        if match:
            trial_num, _, val_loss, gap = match.groups()
            trial_num = int(trial_num)
            val_loss = float(val_loss)
            gap = float(gap)

            if val_loss < val_thresh and gap < gap_thresh:
                if trial_num in optuna_lines:
                    output_lines.append(optuna_lines[trial_num])
                output_lines.append(line)

    with open(output_file, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"Filtered results saved to: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter Optuna trials based on validation loss and overfit gap.")
    parser.add_argument("input_file", help="Path to the .txt file containing trial logs.")
    parser.add_argument("--val_thresh", type=float, default=3e-4, help="Threshold for validation loss (default: 3e-4).")
    parser.add_argument("--gap_thresh", type=float, default=9e-5, help="Threshold for overfit gap (default: 9e-5).")

    args = parser.parse_args()
    filter_trials(args.input_file, args.val_thresh, args.gap_thresh)