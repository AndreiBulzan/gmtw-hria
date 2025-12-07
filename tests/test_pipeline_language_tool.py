import json
import language_tool_python

INPUT_FILE = "data/outputs_8b.jsonl"

SEVERITY_WEIGHTS = {
    "grammar": 3,
    "misspelling": 2,
    "typographical": 1,
    "style": 1,
}


def extract_clean_text(full_output: str) -> str:
    """
    Removes the final JSON itinerary block from the model output.
    Keeps only the narrative text before: `{\n`.
    """
    clean_text, _, _ = full_output.partition("{\n")
    return clean_text.strip()


def compute_score(matches, word_count: int) -> (float, float, float):
    """
    Computes weighted error density and final 0-100 score.
    """
    if word_count == 0:
        return 0, 0, 100.0

    weighted_sum = 0
    for m in matches:
        issue_type = m.rule_issue_type.lower()
        weighted_sum += SEVERITY_WEIGHTS.get(issue_type, 1)

    error_density = (weighted_sum / word_count) * 100
    score = max(0, 100 - error_density)

    return weighted_sum, error_density, score


def main():
    print("Loading LanguageTool...")
    tool = language_tool_python.LanguageTool("ro")  # Romanian

    print(f"Processing file: {INPUT_FILE}\n")

    results = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            entry = json.loads(line)
            instance_id = entry["instance_id"]
            output_text = entry["output"]

            clean_text = extract_clean_text(output_text)
            word_count = len(clean_text.split())

            matches = tool.check(clean_text)
            weighted_errors, density, score = compute_score(matches, word_count)

            results.append({
                "instance_id": instance_id,
                "word_count": word_count,
                "num_matches": len(matches),
                "weighted_errors": weighted_errors,
                "error_density": density,
                "score": score,
            })

            print(f"### Instance: {instance_id}")
            print(f"Words: {word_count}")
            print(f"Matches: {len(matches)}")
            print(f"Weighted errors: {weighted_errors}")
            print(f"Error density: {density:.2f}%")
            print(f"Score: {score:.2f}/100")
            print("-" * 50)

    print("\nDone!")


if __name__ == "__main__":
    main()
