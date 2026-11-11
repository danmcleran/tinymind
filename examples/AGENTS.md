# examples

Repo-wide rules are in the root `AGENTS.md`; each example is described in
`docs/ARCHITECTURE.md` and has a page under `docs/examples/`.

Every runnable example writes a header-row CSV to `output/` and ships a `plot.py` using the shared `examples/plotting/tinymind_plot.py` style module (`make plot`). matplotlib goes in an isolated env (venv/pyenv), never system Python. int8 exemplars expose `make run` / `make bench` / `make golden`.

## Rules

- Every example here is built with the host compiler by `make check` and with `clang++` by `check-clang-examples`, which discovers Makefiles from the tree. Anything that targets Arm only belongs in `benchmarks/`.
- A new example needs: a `README.md`, a `docs/examples/<name>.md` page, an entry in `docs/gallery.md` for its plot, a line in the README project tree, and its `output/` directory in `.gitignore`.
- Plots for the docs site are rendered with `TINYMIND_PLOT_THEME=dark` (the default is light) from the `.venv-plot/` venv and copied to `docs/assets/plots/`. Look at the PNG before committing it: the fit has to be visibly good, not just produced.
- Numbers quoted in a README, a docs page or release notes are re-measured by running the example, never copied from an older write-up.
- **Changing training arithmetic:** re-run every example that trains, master against branch, over 10 seeds (weight seed 1, data seeds 1 to 10; make the seeds overridable in scratch copies of both trees), diff the outputs, and compare mean and standard deviation. Report the default seed and the 10-seed figure together; never choose a flattering seed. A learning rate that was tuned against a since-fixed bug may need retuning (`har_activity` went from 0.05 to 0.1 in #267).
- Fixed-point tabular MLPs: Q16.16 with ReLU and z-scored inputs is the working recipe. Q8.8 trains small problems such as `xor`; check a wider one over several seeds before relying on it.
- LSTM/GRU training wants gradient clipping, Adam and a learning-rate schedule; see `LSTM.md`.
