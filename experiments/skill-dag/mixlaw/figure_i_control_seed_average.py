import json
from pathlib import Path

seed12536 = [
    [0, 4.426135], [125, 2.682668], [250, 2.339392], [375, 2.187189],
    [500, 2.029042], [625, 1.927281], [750, 1.854915], [875, 1.801895],
    [1000, 1.767333], [1125, 1.729474], [1250, 1.717138], [1375, 1.697052],
    [1500, 1.679972], [1625, 1.670788], [1750, 1.650263], [1875, 1.651802],
    [2000, 1.642259], [2125, 1.638649], [2250, 1.635199], [2384, 1.636967],
    # step 2375 (1.637493) deliberately excluded: seed 12345's checkpoint
    # ladder skipped 2375 (near-duplicate of the 2384 final per
    # checkpoint_ladder.py's contract), so it has no matching point to average.
]
seed12345 = [
    [0, 4.47281289100647], [125, 2.6494065046310427], [250, 2.3325371623039244],
    [375, 2.186054158210754], [500, 2.0785145699977874], [625, 1.8904495716094971],
    [750, 1.8403525471687316], [875, 1.7797980070114137], [1000, 1.7549206644296647],
    [1125, 1.7322413355112076], [1250, 1.7179467409849167], [1375, 1.6939303874969482],
    [1500, 1.669490173459053], [1625, 1.6628274232149125], [1750, 1.6691516071558],
    [1875, 1.6417983323335648], [2000, 1.6441879451274872], [2125, 1.6307029217481612],
    [2250, 1.6377677768468857], [2384, 1.6284936755895614],
]

steps_a = [s for s, _ in seed12536]
steps_b = [s for s, _ in seed12345]
assert steps_a == steps_b, (steps_a, steps_b)

averaged = [[s, (va + vb) / 2] for (s, va), (_, vb) in zip(seed12536, seed12345)]

PATH = Path(__file__).resolve().parent / "figure_i_wandb_curves.json"


def main() -> None:
    """Write the averaged control plus both individual seeds into the figure data.

    The two per-seed curves are kept because Figure I draws the control as a
    seed-to-seed band, and that band is the only uncertainty in the figure the
    data supports: the control ran at two seeds, every fitted mixture ran once.
    """
    data = json.loads(PATH.read_text(encoding="utf-8"))
    data["olmo_mix_1124"] = averaged
    data["olmo_mix_1124_seed6198"] = [list(p) for p in seed12536]
    data["olmo_mix_1124_seed12345"] = [list(p) for p in seed12345]
    PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"Wrote {PATH} (control average + 2 seeds)")


if __name__ == "__main__":
    main()
