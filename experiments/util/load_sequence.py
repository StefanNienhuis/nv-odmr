from zhinst.toolkit import Sequence

def load_sequence(file: str) -> Sequence:
    with open(file, "r") as f:
        sequence = Sequence()
        sequence.code = f.read()
        return sequence