import os
import subprocess
import sys

# ─── MASTER CONTROL ROOM ──────────────────────────────────────────────────
# Set any of the following to True to execute that stage of the pipeline.

DRY_RUN = False

PIPELINE_CONFIG = {
    "TESTING": {
        "create_dummy_data.py": False
    },
    "TRAINING": {
        "run_multiseed_optimized.py": False,
        "run_mc_dropout_seed_hybrid_ablation_sensitivity.py": False,
        "run_walk_forward_all_models.py": False,
        "run_retrain_pinn1_only.py": False,
        "run_retrain_pinn2_only.py": False
    },
    "EVALUATION": {

        "add_r2_metrics.py": False,
        "export_r2_table.py": False,
        "evaluate_pinn1_missing.py": False
    },
    "PLOTTING": {
        "generate_loss_curves.py": True,
        "plot_loss_curves_with_physics.py": True,
        "plot_pinn2_comparison.py": True,
        "generate_multiseed_prediction_plot.py": True,
        "generate_review_figures.py": True,
        "generate_all_thesis_figures.py": True,
        "plot_pinn2_monthly_dynamics.py": True,
        "plot_pinn2_10d_dynamics.py": True
    }
}

# ──────────────────────────────────────────────────────────────────────────

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(base_dir, "scripts")

    folders = {
        "TESTING": os.path.join(scripts_dir, "plotting"),
        "TRAINING": os.path.join(scripts_dir, "training"),
        "EVALUATION": os.path.join(scripts_dir, "evaluation"),
        "PLOTTING": os.path.join(scripts_dir, "plotting")
    }

    executed_any = False

    for category, folder in folders.items():
        print(f"\n[{category} STAGE]")
        
        for script_name, should_run in PIPELINE_CONFIG[category].items():
            if should_run:
                executed_any = True
                script_path = os.path.join(folder, script_name)
                
                if not os.path.exists(script_path):
                    print(f"  [ERROR] Cannot find script: {script_path}")
                    continue
                
                print(f"  >>> Running {script_name}...")
                
                try:
                    # Run the script using the current Python executable
                    cmd = [sys.executable, script_path]
                    if category == "TRAINING" and DRY_RUN:
                        cmd.append("--dry-run")
                    
                    env = os.environ.copy()
                    env["PYTHONIOENCODING"] = "utf-8"
                    subprocess.run(cmd, check=True, cwd=os.path.dirname(base_dir), env=env)
                    print(f"  [SUCCESS] {script_name} completed.")
                except subprocess.CalledProcessError as e:
                    print(f"  [FAILED] {script_name} crashed with exit code {e.returncode}.")
                    continue
            else:
                print(f"  ... Skipping {script_name}")

    if not executed_any:
        print("\nAll pipeline toggles are set to False. Nothing was executed.")
        print("To run a script, open master_runner.py and set the target script to True.")
    else:
        print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()
