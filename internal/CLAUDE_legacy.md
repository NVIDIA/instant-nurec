When you work on this code base, you must adhere to the following rules:

0. Before you do any implementation or fix in this repo, always check how it's done in the nre repo @/storage/projects/nre
1. We want full branch coverage testing on every function, where possible
2. For every new function in this repo add full branch coverage testing, where possible
3. Do test-driven-development. ALWAYS!
4. When you execute an ent-to-end test, do it the following:
    1. Iterative bring-up on real data.

        Run both, merge and no-merge commands

        1. These are commands which you are allowed to run unsandboxed, as you need a gpu. All calls that require a GPU, are allowed to be run unsandboxed. Only calls that require a GPU are allowed to be run unsandboxed.
        2. Run the commands with a **60 s no-progress watchdog**: if no new log line in 60 s, cancel.
            1. **Stalls** → add `logger.info` at the suspected stall point, commit `chore(debug): log <step>`, re-run.
            2. **Fails with traceback** → look up the equivalent in `/storage/projects/nre`, port the fix, commit: `fix(<area>): <one-liner referencing NRE source>`. Self-invented fix only if NRE has no equivalent (e.g. we removed `pytorch_lightning`); commit message must say `(self-invented: <reason>)`.
            3. Go back to 4.1.
        3. Repeat until both `--merge none` and `--merge frustum-ownership` modes finish without an error..
    2. Baseline parity loop
        1. Ask for a baseline to compare against, it should be part of the plan!
        2. Add `scripts/validate_parity.py`:

            ```python
            # Loads two PLYs with plyfile, asserts:
            #   - same number of files. This has to be exact, no if else or buts
            #   - same vertex count. This has to be exact, no if else or buts
            #   - same property names + dtypes. This has to be exact, no if else or buts
            #   - per-property |a - b| < 1e-3 across all elements
            ```

        3. Run three comparisons:
            For all necessary comparison to validate parity on the ply output
            For comparison do:
            1. **Wrong file count** → fix  (same as in 4.1.2.2.), commit (same as in 4.1.2.2.), re-run, go back to 4.1.
            2. **Wrong PLY count** → fix  (same as in 4.1.2.2.), commit (same as in 4.1.2.2.), re-run, go back to 4.1.
            3. **Largely different file size** → fix  (same as in 4.1.2.2.), commit (same as in 4.1.2.2.), re-run, go back to 4.1.
            4. **Schema mismatch** → fix  (same as in 4.1.2.2.), commit (same as in 4.1.2.2.), re-run, go back to 4.1.
            5. **Per-element diff > 1e-3** → fix  (same as in 4.1.2.2.), commit (same as in 4.1.2.2.), re-run, go back to 4.1.
            6. All three comparisons green → exit loop.

Each fix is a separate commit so the bring-up history shows how parity was reached.