def process_text(row: dict) -> str:
    """A mock function to simulate processing text."""
    input_text = row.get("text", "")
    return f"Processed: ||| {input_text} |||"


def score_text(row: dict) -> float:
    """A mock scorer: the length of the text, for driving a filter step."""
    return float(len(row.get("text", "")))
