Build a /plan from this prompt:

We want to make one entrypoint of the @/storage/projects/nre repo completely standalone in this empty repository. The entrypoint of interest is the nrm kelvin model predict mode, see @nre_example_call.sh for the only two code paths this repo should have. To achieve this, adhere to the @CLAUDE.md file and follow the steps layed out in this plan:

# Phase 0 - Initial setup, comparison and parity:

0. Generated baselines: I ran the script @nre_example_call.sh from within the repo @/storage/projects/nre on the commit a54a6af0a177beabd01fe37e398c45be165a270f, and generated ply, parsed.yaml and log.txt files for both, the merge and no_merge case. Those are in @baselines/original_baselines
1. Write a script that can be used to validate parity between ply files. The script should take as input the baseline ply files and proposed ply files. The script should compare them like this:
   1. Compare the number of files. This has to be exact, no if else or buts.
   2. Compare the number of vertices. This has to be exact, no if else or buts.
   3. Compare the properties names and dtypes. This has to be exact, no if else or buts.
   4. Compare the per-element differences. This has to be less than 1e-3, no if else or buts.
   5. If all comparisons are successful, the script should exit with status 0. If any comparison fails, the script should exit with status 1 and print the details of the failed comparison.
   6. The script should be called with the following arguments @compare_ply.py merge <baseline_merge_ply_file> <proposed_merge_ply_file>, as well as @compare_ply.py no_merge <path/to/baseline_no_merge_ply_dir> <path/to/proposed_no_merge_ply_dir>.
   7. Run the script on the baselines I created in step 0. For the proposed ply use the baseline ply files, this should compare identical files and act as a test.
   8. I generated 5 more baselines, they are in @baselines/more_baselines. Run the parity script on them. This determines if the code in @/storage/projects/nre repo is deterministic. If it's not deterministic, adjust the script to account for this, ie. allow for small tolerances, defined by the non-determinism of the repeated baseline plys generations.
2. Copy the whole code: From the @/storage/projects/nre repo on the commit a54a6af0a177beabd01fe37e398c45be165a270f copy the whole code to this repository.
   1. Rerun @nre_example_call.sh, but this time run it from within this repo/directory.
   2. Compare the ply files from 2.1. to the ones from 0. This should be within the same tolerance as determined in 1.8. If it's not, find out why and fix it.

# Phase 1 - Build minimal codebase:

**Note:** Always ensure parity - at each step! From this step onwards, always keep parity with the baselines AT EACH STEP and SUBSTEP of this plan using your script you wrote in step 1. Look at the @CLAUDE.md file how this is done!

3. Create minimal CLI interface: Create a new entrypoint that only has the following parameters --ncore-path /storage/data/nurec/ncorev4 --output-dir /tmp/nurec_iter/no_merge --merge [none,frustum-ownership] --log-level [INFO,DEBUG,...]
   1. predict.primitive_merge.enabled=false maps to --merge none (should be default in this code base) and the (predict.primitive_merge.enabled=true predict.primitive_merge.overlap_strategy=frustum_ownership) maps to --merge frustum-ownership
4. Strip boilerplate files/functions/lines of code: The full copied codebase, has way too much code for our purpose of ONLY having the kelvin model predict mode as standalone in this code base. Be drastic, be aggressive, change the structure, change the files, change the code to simplify the codebase as much as possible, while ensuring parity with the baseline. Every commit you make during this phase should list the number of files/LOC at start, how many were removed during its commit, and how many are left in the codebase.
   0. Iteratively delete all files of all types that are not necessary for this repository.
   1. Iteratively delete all files of all types and all functions that are not necessary for nrm.
   2. Iteratively strip all files of all types and all functions that are not necessary for the kelvin model.
   3. Iteratively strip all files of all types and all functions that are not necessary for the kelvin model predict mode. That includes, but is not limited to removing everything related to training, testing, validation. Use the baseline's created config/parsed.yaml file to understand the limited functionality needed for the predict mode.
   4. Iteratively strip all files of all types and all functions and all lines of code that do not change the output of the kelvin model predict mode, ie if a code piece does not alter the final ply file, remove it. That includes, but is not limited to profiling code - it might be executed, but has no impacton the final ply file. Understand the full codebase in detail to to realize this step.
   5. Iteratively remove all yaml files too, the final version should not contain, nor depend on any yaml file.
   6. Iteratively remove NRE as a dependency, we only need the nrm kelvin model predict mode.
   7. Iteratively remove NRE mentioned anywhere in the codebase.
   8. Ensure minimal code and /simplify as much as possible! Final code should only contain everything related to handling the ncorev4 data, to prepare a batch, to run the kelvin model predict mode, and to export the ply files. No other code is allowed. If that's not true, go back to the beginning of this step.
   9. Iterate over this phase until you can't strip any further and a single file/function removal would cause a loss of parity for a given ncorev4 sample. Use your /effort max on this phase and understand the codebase in detail to do this!
5. Save/Load entire model: Use the approach of saving and loading the FULL kelvin model, see https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model.
   1. After the full-model pickle generation, go back to step 4. Iterate over steps 4. and 5. until convergence.
6. Unit tests: Write unit tests for branch coverage on every function that is left in the codebase.

# Phase 2 - Strip CUDA/slang:

**Note:** Always ensure parity - at each step! From this step onwards, always keep parity with the baselines AT EACH STEP and SUBSTEP of this plan using your script you wrote in step 1. Look at the @CLAUDE.md file how this is done!

7. Iteratively remove dependencies on CUDA/slang using test-driven-development: This repo should be a pure python/pytorch implementation.
   0. Replace all CUDA dependencies including external libraries (nvdiffrast, torch_scatter, gsplat) with pure-PyTorch equivalents, iff they're in the code path of the Kelvin model predict modez.
   1. Implement your own versions of the CUDA/slang code, verify with dedicated tests that both implementations are equivalent up to machine accuracy. This is crucial! Build tests that verify every code path of both implementations is equivalent for each input, don't forget edge cases. 
   2. Once verified CUDA/slang and your pytorch implemenations are equivalent, verify parity of the ply files still hold. Take the verified difference between your pytorch and CUDA/slang implementations into account and adjust the tolerance of the parity accordingly: TOL_new = max(TOL_old, max_observed_diff_in_equivalence_test). Document every TOL revision in tests/tolerance.json.
   3. This phase must leave only torch's CUDA.
   4. Repeat Phase 1 on the cuda/slang stripped codebase.

# Phase 3 - Change build system and remove 3rd party dependencies:

**Note:** Always ensure parity - at each step! From this step onwards, always keep parity with the baselines AT EACH STEP and SUBSTEP of this plan using your script you wrote in step 1. Look at the @CLAUDE.md file how this is done!

8. Iteratively remove dependencies, build similar setup and run script, and make structure similar to the @NVIDIA/asset-harvester repo: https://github.com/NVIDIA/asset-harvester/tree/main
   1. The dependencies have to exclude bazel, pytorch lightning, hydra, nre and other dependencies that are not necessary for the instant-nurec project. The dependencies should be minimal and similar to the dependencies of the @NVIDIA/asset-harvester repo.
   2. Create a repo structure that is very similar to the @NVIDIA/asset-harvester repo - understand it fully, each directory, each file, each line of code of that repo
   3. Move all files and directories that were part of the migration effort into in internal directory. The code should still work, if the interal directory has been deleted.
   4. Repeat Phase 1 on the build system and dependency stripped codebase.

# Phase 4 - Cleanup and final touches:

**Note:** Always ensure parity - at each step! From this step onwards, always keep parity with the baselines AT EACH STEP and SUBSTEP of this plan using your script you wrote in step 1. Look at the @CLAUDE.md file how this is done!

9. Mock huggingface: Assume the ncorev4 data and the kelvin model is being downloaded from huggingface, create mocks for this. Use the already available local model and data to for this, ie the huggingface mocks should resolve to the local model and data, because it's not yet on huggingface. This is being done by the corporation, not by us.
10. README, setup, run: Write a README.md, setup.sh, run.sh file, similar to the @NVIDIA/asset-harvester repo README.md, setup.sh, and run.sh file. The readme should be self-contained and explain how to use the codebase. The setup.sh should be self-contained and setup the environment. The run.sh should be self-contained and enable running the codebase.



**Notes:**
1. Always ensure parity - at each step! From this step onwards, always keep parity with the baselines AT EACH STEP and SUBSTEP of this plan using your script you wrote in step 1. Look at the @CLAUDE.md file how this is done! Parity must be kept at all times, up to tolerance you determined in the above steps!
2. All calls that require a GPU, are allowed to be run unsandboxed. Only calls that require a GPU are allowed to be run unsandboxed.
3. Work completely autonomously and do not ask for permission, until you're done with ALL tasks in this plan and created a final MR and reached parity. If you hit a blocker resolve it yourself and continue with the plan.
4. Work on a single branch for this plan. Do not create any other branches. Create a fresh branch off main. Do not reuse existing port branches.
5. Each step on its own can and should be an iterative step
6. Create a clean commit with a descriptive commit message for each step and each substep.
7. Phase 1 is by far the most important step, as it builds the minimal codebase.

Use the AskUserQuestion tool to ask any question that needs clarification. No ambiguities should be left. Also. Let me know how to change my prompt, such that this question does not arise.