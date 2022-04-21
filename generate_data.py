with open("alemtuzumab_sequences.txt", "r") as f:
    sequences = [seq.split("\n") for seq in f.read().split(">") if seq]
    sequences = ["".join(parts).strip() for _, *parts in sequences]

    target_sequence = sequences[1]
    missing_indices = set([0, 1, 2, 24, 25, 26, 62, 63, 64, 66, 67, 68, 69, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 209, 210, 211, 212, 213])

    seq_with_gaps = "".join(c for i, c in enumerate(target_sequence) if i not in missing_indices)

    with open("light_chain_with_gaps.txt", "w+") as output_file:
        output_file.writelines([
            "UNKNOWN SEQUENCE WITH GAPS",
            seq_with_gaps
        ])
