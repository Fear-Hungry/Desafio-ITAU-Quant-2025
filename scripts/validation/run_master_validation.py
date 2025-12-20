#!/usr/bin/env python3
"""Master orchestration script for comprehensive PRISM-R validation.

This script coordinates the complete validation pipeline:
1. Data pipeline (fresh download or cached)
2. Full backtest on primary configs
3. Baseline comparisons (1/N, MV, RP)
4. Sensitivity analyses (cost, window, covariance)
5. Validation test suite
6. Results aggregation
7. Report generation

Usage:
    # Full validation (20-30 min)
    poetry run python scripts/validation/run_master_validation.py --mode full

    # Quick smoke test (5 min)
    poetry run python scripts/validation/run_master_validation.py --mode quick --skip-download

    # Production pre-deploy validation (10 min)
    poetry run python scripts/validation/run_master_validation.py --mode production

    # Resume from specific stage
    poetry run python scripts/validation/run_master_validation.py --resume-from stage3

Author: PRISM-R Team
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from arara_quant.config import get_settings
from arara_quant.reports.canonical import ensure_output_dirs

# Project root
SETTINGS = get_settings()
ensure_output_dirs(SETTINGS)

PROJECT_ROOT = SETTINGS.project_root
RESULTS_DIR = SETTINGS.results_dir
REPORTS_DIR = SETTINGS.reports_dir
CONFIGS_DIR = SETTINGS.configs_dir

# Validation configurations
PRIMARY_CONFIGS = [
    "configs/optimizer_example.yaml",
    "configs/optimizer_regime_aware.yaml",
    "configs/optimizer_adaptive_hedge.yaml",
]

SENSITIVITY_SCRIPTS = [
    "scripts/research/run_cost_sensitivity.py",
    "scripts/research/run_window_sensitivity.py",
    "scripts/research/run_covariance_sensitivity.py",
]

VALIDATION_SCRIPTS = [
    "scripts/validation/run_comprehensive_tests.py",
    "scripts/validation/run_constraint_tests.py",
    "scripts/validation/run_estimator_tests.py",
    "scripts/validation/run_sensitivity_tests.py",
]


class ValidationOrchestrator:
    """Master orchestrator for PRISM-R validation pipeline."""

    def __init__(
        self,
        mode: str = "full",
        output_dir: Optional[Path] = None,
        skip_download: bool = False,
        skip_sensitivity: bool = False,
        skip_validation: bool = False,
        resume_from: Optional[str] = None,
        parallel: bool = False,
    ):
        """Initialize orchestrator.

        Args:
            mode: Execution mode ('quick', 'full', 'production')
            output_dir: Output directory (default: outputs/reports/validation_YYYYMMDD_HHMMSS)
            skip_download: Skip data download (use cached data)
            skip_sensitivity: Skip sensitivity analyses
            skip_validation: Skip validation test suite
            resume_from: Resume from specific stage (e.g., 'stage3')
            parallel: Run independent analyses in parallel
        """
        self.mode = mode
        self.skip_download = skip_download
        self.skip_sensitivity = skip_sensitivity
        self.skip_validation = skip_validation
        self.resume_from = resume_from
        self.parallel = parallel

        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or REPORTS_DIR / f"validation_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Execution log
        self.execution_log: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []

        # Stage completion tracking
        self.completed_stages: set = set()

        print(f"=== PRISM-R Master Validation Orchestrator ===")
        print(f"Mode: {mode}")
        print(f"Output: {self.output_dir}")
        print(f"Skip download: {skip_download}")
        print(f"Skip sensitivity: {skip_sensitivity}")
        print(f"Skip validation: {skip_validation}")
        print(f"Resume from: {resume_from or 'beginning'}")
        print("=" * 50)

    def run_command(
        self,
        cmd: List[str],
        stage: str,
        description: str,
        capture_output: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Execute shell command with logging.

        Args:
            cmd: Command to execute (list of strings)
            stage: Stage identifier
            description: Human-readable description
            capture_output: Capture stdout/stderr

        Returns:
            (success, output) tuple
        """
        print(f"\n[{stage}] {description}")
        print(f"Command: {' '.join(cmd)}")

        start_time = datetime.now()

        try:
            if capture_output:
                result = subprocess.run(
                    cmd,
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout
                )
                success = result.returncode == 0
                output = result.stdout if success else result.stderr
            else:
                result = subprocess.run(
                    cmd,
                    cwd=PROJECT_ROOT,
                    timeout=3600,
                )
                success = result.returncode == 0
                output = None

            duration = (datetime.now() - start_time).total_seconds()

            log_entry = {
                "stage": stage,
                "description": description,
                "command": " ".join(cmd),
                "success": success,
                "duration_sec": duration,
                "timestamp": start_time.isoformat(),
            }

            self.execution_log.append(log_entry)

            if success:
                print(f"✓ Success ({duration:.1f}s)")
            else:
                print(f"✗ Failed ({duration:.1f}s)")
                self.errors.append(
                    {
                        "stage": stage,
                        "description": description,
                        "error": output or "Command failed",
                    }
                )

            return success, output

        except subprocess.TimeoutExpired:
            print(f"✗ Timeout (>3600s)")
            self.errors.append(
                {
                    "stage": stage,
                    "description": description,
                    "error": "Command timeout (>1 hour)",
                }
            )
            return False, None

        except Exception as e:
            print(f"✗ Exception: {e}")
            self.errors.append(
                {
                    "stage": stage,
                    "description": description,
                    "error": str(e),
                }
            )
            return False, None

    def should_run_stage(self, stage: str) -> bool:
        """Check if stage should be executed."""
        if self.resume_from:
            # Extract stage number
            stage_num = int(stage.replace("stage", ""))
            resume_num = int(self.resume_from.replace("stage", ""))
            return stage_num >= resume_num
        return True

    def stage1_data_pipeline(self) -> bool:
        """Stage 1: Data acquisition and processing."""
        if not self.should_run_stage("stage1"):
            print("\n[STAGE 1] Skipping (resume mode)")
            return True

        print("\n" + "=" * 50)
        print("STAGE 1: Data Pipeline")
        print("=" * 50)

        if self.skip_download:
            print("Skipping data download (using cached data)")
            return True

        # Run data pipeline script
        cmd = [
            "poetry",
            "run",
            "python",
            "scripts/core/run_01_data_pipeline.py",
            "--force-download",
            "--start",
            "2010-01-01",
        ]

        success, _ = self.run_command(
            cmd,
            "stage1",
            "Download and process market data",
        )

        self.completed_stages.add("stage1")
        return success

    def stage2_primary_backtests(self) -> bool:
        """Stage 2: Run backtests on primary configurations."""
        if not self.should_run_stage("stage2"):
            print("\n[STAGE 2] Skipping (resume mode)")
            return True

        print("\n" + "=" * 50)
        print("STAGE 2: Primary Backtests")
        print("=" * 50)

        # Select configs based on mode
        if self.mode == "quick":
            configs = [PRIMARY_CONFIGS[0]]  # Only standard config
        elif self.mode == "production":
            configs = ["configs/production_erc_v2.yaml"]
        else:  # full
            configs = PRIMARY_CONFIGS

        all_success = True

        for config in configs:
            config_name = Path(config).stem
            output_file = self.output_dir / f"backtest_{config_name}.json"

            cmd = [
                "poetry",
                "run",
                "arara-quant",
                "backtest",
                "--config",
                config,
                "--no-dry-run",
                "--wf-report",
                "--json",
            ]

            success, output = self.run_command(
                cmd,
                "stage2",
                f"Backtest: {config_name}",
                capture_output=True,
            )

            if success and output:
                output_file.write_text(output)

            all_success = all_success and success

        self.completed_stages.add("stage2")
        return all_success

    def stage3_baseline_comparisons(self) -> bool:
        """Stage 3: Compare against baseline strategies."""
        if not self.should_run_stage("stage3"):
            print("\n[STAGE 3] Skipping (resume mode)")
            return True

        print("\n" + "=" * 50)
        print("STAGE 3: Baseline Comparisons")
        print("=" * 50)

        # Run baseline comparison via CLI
        cmd = [
            "poetry",
            "run",
            "arara-quant",
            "compare-baselines",
        ]

        success, _ = self.run_command(
            cmd,
            "stage3",
            "Compare 1/N, MV, RP strategies",
        )

        self.completed_stages.add("stage3")
        return success

    def stage4_sensitivity_analyses(self) -> bool:
        """Stage 4: Run sensitivity analyses."""
        if not self.should_run_stage("stage4"):
            print("\n[STAGE 4] Skipping (resume mode)")
            return True

        print("\n" + "=" * 50)
        print("STAGE 4: Sensitivity Analyses")
        print("=" * 50)

        if self.skip_sensitivity or self.mode == "quick":
            print("Skipping sensitivity analyses")
            return True

        all_success = True

        for script in SENSITIVITY_SCRIPTS:
            script_name = Path(script).stem

            cmd = [
                "poetry",
                "run",
                "python",
                script,
            ]

            success, _ = self.run_command(
                cmd,
                "stage4",
                f"Sensitivity: {script_name}",
            )

            all_success = all_success and success

        self.completed_stages.add("stage4")
        return all_success

    def stage5_validation_suite(self) -> bool:
        """Stage 5: Run comprehensive validation tests."""
        if not self.should_run_stage("stage5"):
            print("\n[STAGE 5] Skipping (resume mode)")
            return True

        print("\n" + "=" * 50)
        print("STAGE 5: Validation Test Suite")
        print("=" * 50)

        if self.skip_validation or self.mode == "quick":
            print("Skipping validation suite")
            return True

        all_success = True

        for script in VALIDATION_SCRIPTS:
            script_name = Path(script).stem

            cmd = [
                "poetry",
                "run",
                "python",
                script,
            ]

            success, _ = self.run_command(
                cmd,
                "stage5",
                f"Validation: {script_name}",
            )

            all_success = all_success and success

        self.completed_stages.add("stage5")
        return all_success

    def stage6_aggregate_results(self) -> bool:
        """Stage 6: Aggregate all results into master tables."""
        if not self.should_run_stage("stage6"):
            print("\n[STAGE 6] Skipping (resume mode)")
            return True

        print("\n" + "=" * 50)
        print("STAGE 6: Results Aggregation")
        print("=" * 50)

        try:
            # Collect backtest results
            backtest_results = []
            for json_file in self.output_dir.glob("backtest_*.json"):
                with open(json_file) as f:
                    data = json.load(f)
                    backtest_results.append(data)

            if backtest_results:
                backtest_df = pd.DataFrame(backtest_results)
                output_file = self.output_dir / "master_backtest_results.csv"
                backtest_df.to_csv(output_file, index=False)
                print(f"✓ Saved backtest results: {output_file}")

            # Collect baseline comparison results
            baseline_files = list(RESULTS_DIR.glob("baselines/*.csv"))
            if baseline_files:
                baseline_dfs = [pd.read_csv(f) for f in baseline_files]
                baseline_df = pd.concat(baseline_dfs, ignore_index=True)
                output_file = self.output_dir / "master_baseline_results.csv"
                baseline_df.to_csv(output_file, index=False)
                print(f"✓ Saved baseline results: {output_file}")

            # Collect sensitivity results
            sensitivity_dirs = [
                "cost_sensitivity",
                "window_sensitivity",
                "cov_sensitivity",
            ]
            for sens_dir in sensitivity_dirs:
                sens_path = RESULTS_DIR / sens_dir
                if sens_path.exists():
                    csv_files = list(sens_path.glob("*.csv"))
                    if csv_files:
                        sens_dfs = [pd.read_csv(f) for f in csv_files]
                        sens_df = pd.concat(sens_dfs, ignore_index=True)
                        output_file = self.output_dir / f"master_{sens_dir}.csv"
                        sens_df.to_csv(output_file, index=False)
                        print(f"✓ Saved {sens_dir} results: {output_file}")

            self.completed_stages.add("stage6")
            return True

        except Exception as e:
            print(f"✗ Aggregation failed: {e}")
            self.errors.append(
                {
                    "stage": "stage6",
                    "description": "Results aggregation",
                    "error": str(e),
                }
            )
            return False

    def stage7_generate_report(self) -> bool:
        """Stage 7: Generate executive summary report."""
        if not self.should_run_stage("stage7"):
            print("\n[STAGE 7] Skipping (resume mode)")
            return True

        print("\n" + "=" * 50)
        print("STAGE 7: Report Generation")
        print("=" * 50)

        try:
            report_path = self.output_dir / "VALIDATION_SUMMARY.md"

            with open(report_path, "w") as f:
                f.write("# PRISM-R Validation Summary\n\n")
                f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
                f.write(f"**Mode:** {self.mode}\n\n")
                f.write(f"**Completed Stages:** {sorted(self.completed_stages)}\n\n")

                f.write("## Execution Summary\n\n")
                f.write(f"- Total commands: {len(self.execution_log)}\n")
                f.write(
                    f"- Successful: {sum(1 for e in self.execution_log if e['success'])}\n"
                )
                f.write(
                    f"- Failed: {sum(1 for e in self.execution_log if not e['success'])}\n"
                )
                total_time = sum(e["duration_sec"] for e in self.execution_log)
                f.write(f"- Total runtime: {total_time / 60:.1f} minutes\n\n")

                if self.errors:
                    f.write("## Errors\n\n")
                    for err in self.errors:
                        f.write(f"### {err['stage']}: {err['description']}\n\n")
                        f.write(f"```\n{err['error']}\n```\n\n")

                f.write("## Results Files\n\n")
                for result_file in sorted(self.output_dir.glob("master_*.csv")):
                    f.write(f"- `{result_file.name}`\n")

                f.write("\n## Next Steps\n\n")
                f.write("1. Review backtest metrics in `master_backtest_results.csv`\n")
                f.write("2. Compare against baselines in `master_baseline_results.csv`\n")
                f.write("3. Check sensitivity analyses for robustness\n")
                f.write("4. Update README.md with final results\n")

            print(f"✓ Generated report: {report_path}")

            # Save execution log
            log_path = self.output_dir / "execution_log.json"
            with open(log_path, "w") as f:
                json.dump(self.execution_log, f, indent=2)
            print(f"✓ Saved execution log: {log_path}")

            self.completed_stages.add("stage7")
            return True

        except Exception as e:
            print(f"✗ Report generation failed: {e}")
            self.errors.append(
                {
                    "stage": "stage7",
                    "description": "Report generation",
                    "error": str(e),
                }
            )
            return False

    def run(self) -> bool:
        """Execute the complete validation pipeline."""
        start_time = datetime.now()

        print(f"\nStarting validation pipeline at {start_time.isoformat()}\n")

        # Execute all stages
        stages = [
            ("stage1", self.stage1_data_pipeline),
            ("stage2", self.stage2_primary_backtests),
            ("stage3", self.stage3_baseline_comparisons),
            ("stage4", self.stage4_sensitivity_analyses),
            ("stage5", self.stage5_validation_suite),
            ("stage6", self.stage6_aggregate_results),
            ("stage7", self.stage7_generate_report),
        ]

        for stage_name, stage_func in stages:
            success = stage_func()
            if not success and self.mode != "full":
                # In quick/production mode, stop on failure
                print(f"\n✗ Stage failed: {stage_name}")
                print("Stopping pipeline (use --mode full to continue on errors)")
                break

        # Final summary
        duration = datetime.now() - start_time
        print("\n" + "=" * 50)
        print("VALIDATION PIPELINE COMPLETE")
        print("=" * 50)
        print(f"Total runtime: {duration.total_seconds() / 60:.1f} minutes")
        print(f"Completed stages: {sorted(self.completed_stages)}")
        print(f"Errors: {len(self.errors)}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 50)

        return len(self.errors) == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PRISM-R Master Validation Orchestrator"
    )

    parser.add_argument(
        "--mode",
        choices=["quick", "full", "production"],
        default="full",
        help="Execution mode (default: full)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: outputs/reports/validation_YYYYMMDD_HHMMSS)",
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download (use cached data)",
    )

    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip sensitivity analyses",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation test suite",
    )

    parser.add_argument(
        "--resume-from",
        choices=["stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"],
        help="Resume from specific stage",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run independent analyses in parallel (experimental)",
    )

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = ValidationOrchestrator(
        mode=args.mode,
        output_dir=args.output_dir,
        skip_download=args.skip_download,
        skip_sensitivity=args.skip_sensitivity,
        skip_validation=args.skip_validation,
        resume_from=args.resume_from,
        parallel=args.parallel,
    )

    # Run pipeline
    success = orchestrator.run()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
