# Examples

Run from the repository root. Each script runs on CPU and shows a complete backward pass.
`uv run` installs the project's environment as needed.

| Script | Demonstrates | Command |
| --- | --- | --- |
| [matching.py](matching.py) | Dense predictions, list targets, an empty sample, matching, and ProbIoU | `uv run examples/matching.py` |
| [classification.py](classification.py) | VFL and QFL with caller-supplied quality targets | `uv run examples/classification.py` |
| [custom_loss.py](custom_loss.py) | A custom pure function combined with ProbIoU through the builder | `uv run examples/custom_loss.py` |

`just examples` runs all three scripts. They use fixed inputs and typed error handling;
they do not generate quality targets or orchestrate detector layers.
