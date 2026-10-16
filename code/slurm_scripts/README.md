# SLURM scripts

Job scripts used to run fine-tuning, generation and evaluation on the University of Sheffield Stanage HPC cluster (NVIDIA A100 GPUs). Each script loads the cluster's Python and CUDA modules, activates the MatterGen environment in `$HOME/mattergen_project/mattergen`, and writes outputs to `/mnt/parscratch/users/$USER/mattergen_outputs/`.

Submit a script with `sbatch <script>.sbatch`, from a folder that contains a `logs/` directory.

| Script | What it runs |
|--------|--------------|
| `finetune.sbatch` | Fine-tunes the pre-trained base model with a ΔS (magnetocaloric entropy change) property adapter, 1,000 epochs on 1× A100. |
| `finetune_joint.sbatch` | Joint ΔS + Curie temperature fine-tune, with `dropout_fields_iid=true` so either property can be conditioned on alone. |
| `generate_evaluate.sbatch` | Generates and evaluates every run in the thesis: the unconditional base model (1,008 structures), five ΔS levels, joint ΔS + T<sub>C</sub> at 300 K, five T<sub>C</sub>-only levels, and five superconductor T<sub>c</sub> levels. It logs each step, checks the reference dataset first, and carries on if a single run fails. |
| `evaluate_only.sbatch` | Re-runs evaluation on existing generated structures, giving each evaluation its own temporary folder. |
| `eval_all.sbatch` | Re-runs evaluation by pointing `mattergen-evaluate` directly at each `generated_crystals.extxyz` file. |
| `gen_targeted.sbatch` | Generates candidates for specific applications: ΔS = 32.5 J kg⁻¹ K⁻¹ with T<sub>C</sub> = 3.15 K (cryogenic cooling), 270 K (freezers) and 300 K (air conditioning). |

All evaluation uses MatterSim relaxation and the TRI2024 energy correction against MatterGen's Alex-MP reference dataset.

The pre-training and superconductor fine-tuning scripts aren't included here. The resolved configurations for those runs are in `configs/`.
