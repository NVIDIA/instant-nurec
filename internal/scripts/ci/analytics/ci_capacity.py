#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import datetime
import json
import os

from collections import defaultdict
from typing import Any, Dict, List


class CapacityConfig:
    """Configuration management for capacity analysis"""

    def __init__(self):
        self.non_production_periods = {}
        self.runner_production_periods = {}  # Computed from actual job data

        # Always load from hardcoded path in script's folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "ci_runner_config.json")

        # Config file is required
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self.load_config(config_path)

    def load_config(self, config_file: str) -> None:
        """Load configuration from JSON file"""
        try:
            with open(config_file, "r") as f:
                self.non_production_periods = json.load(f) or {}
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
            self.non_production_periods = {}

    def compute_production_periods(self, job_data: List[Dict]) -> None:
        """Compute production periods for each runner based on actual job data"""
        self.runner_production_periods = {}

        # Group jobs by runner
        jobs_by_runner = defaultdict(list)
        for job in job_data:
            runner_id = self._get_runner_id_from_job(job)
            jobs_by_runner[runner_id].append(job)

        # Compute production period for each runner
        for runner_id, jobs in jobs_by_runner.items():
            if not jobs:
                continue

            # Sort jobs by started_at to find first
            sorted_jobs = sorted(jobs, key=lambda j: j.get("started_at", ""))

            try:
                # Get first job date (convert to UTC)
                first_job = sorted_jobs[0]
                first_started_at = first_job["started_at"]
                if first_started_at.endswith("Z"):
                    first_started_at = first_started_at[:-1] + "+00:00"
                first_date = datetime.datetime.fromisoformat(first_started_at).astimezone(datetime.timezone.utc).date()

                self.runner_production_periods[runner_id] = {
                    "start_date": first_date.isoformat(),
                    "end_date": None,  # Keep runners in production unless explicitly configured otherwise
                }

            except (ValueError, KeyError, TypeError) as e:
                # If we can't parse dates, fail explicitly to identify data quality issues
                raise ValueError(f"Failed to parse job dates for runner '{runner_id}': {e}") from e

    def _get_runner_id_from_job(self, job: Dict) -> str:
        """Get runner identifier from job data (processed format from ci_visualization.py)"""
        return job.get("runner", "unknown-no-id")

    def is_runner_active(self, runner_id: str, date: datetime.date) -> bool:
        """Check if runner was active on given date"""
        # First check if we have computed production periods for this runner
        if runner_id in self.runner_production_periods:
            prod_period = self.runner_production_periods[runner_id]
            start_date = datetime.date.fromisoformat(prod_period["start_date"])
            end_date = prod_period["end_date"]

            # Check if date is within the production period
            # If end_date is None, runner stays in production indefinitely
            date_in_production = date >= start_date
            if end_date is not None:
                end_date_parsed = datetime.date.fromisoformat(end_date)
                date_in_production = start_date <= date <= end_date_parsed

            if date_in_production:
                # Now check if this date falls within any non-production periods
                if runner_id in self.non_production_periods.get("non_production_periods", {}).get("runners", {}):
                    runner_config = self.non_production_periods["non_production_periods"]["runners"][runner_id]
                    periods = runner_config.get("periods", [])

                    for period in periods:
                        # Check start_date (if specified)
                        if "start_date" in period:
                            period_start = datetime.date.fromisoformat(period["start_date"])
                            if date < period_start:
                                continue  # Try next period

                        # Check end_date (if specified)
                        if "end_date" in period:
                            period_end = datetime.date.fromisoformat(period["end_date"])
                            if date > period_end:
                                continue  # Try next period

                        # Date falls within this non-production period
                        return False

                # Date is within production period and not in any non-production period
                return True
            else:
                # Date is outside the production period
                return False

        # If no computed production period, check if there's a config entry
        if runner_id in self.non_production_periods.get("non_production_periods", {}).get("runners", {}):
            runner_config = self.non_production_periods["non_production_periods"]["runners"][runner_id]
            periods = runner_config.get("periods", [])

            for period in periods:
                # Check start_date (if specified)
                if "start_date" in period:
                    start_date = datetime.date.fromisoformat(period["start_date"])
                    if date < start_date:
                        continue  # Try next period

                # Check end_date (if specified)
                if "end_date" in period:
                    end_date = datetime.date.fromisoformat(period["end_date"])
                    if date > end_date:
                        continue  # Try next period

                # Date falls within this non-production period
                return False

            # Date doesn't fall within any non-production period
            return True

        # No production period computed and no config entry - assume active
        return True


class CapacityAnalyzer:
    """Main capacity analysis class"""

    def __init__(self):
        self.config = CapacityConfig()
        self.production_jobs = None
        self.computed_metrics = None

    def filter_production_jobs(self, job_data: List[Dict]) -> List[Dict]:
        """Filter jobs based on non-production periods configuration"""

        # Validate that all jobs have required fields
        invalid_jobs = [i for i, job in enumerate(job_data) if job.get("started_at") is None]
        if invalid_jobs:
            raise ValueError(
                f"Found {len(invalid_jobs)} jobs with None started_at. "
                f"This indicates a data quality issue. Jobs should be preprocessed "
                f"to filter out jobs that never started. First invalid job index: {invalid_jobs[0]}"
            )

        # First compute production periods from actual job data
        self.config.compute_production_periods(job_data)

        if not self.config.non_production_periods and not self.config.runner_production_periods:
            return job_data  # No filtering if no config and no computed periods

        filtered_jobs = []
        discarded_jobs = []

        for job in job_data:
            if not self.is_job_in_non_production_period(job):
                filtered_jobs.append(job)
            else:
                discarded_jobs.append(job)

        return filtered_jobs

    def is_job_in_non_production_period(self, job: Dict) -> bool:
        """Check if job falls within non-production period for its runner"""
        # Get runner identifier (works with both formats)
        runner_id = self._get_runner_id(job)

        # Parse job date (processed format from ci_visualization.py) and convert to UTC
        try:
            started_at = job["started_at"]
            if started_at.endswith("Z"):
                started_at = started_at[:-1] + "+00:00"
            started_at_dt = datetime.datetime.fromisoformat(started_at)
            job_date = started_at_dt.astimezone(datetime.timezone.utc).date()
        except (ValueError, KeyError):
            # If we can't parse the date, include the job
            return False

        # Check if runner is active on this date
        if not self.config.is_runner_active(runner_id, job_date):
            return True  # Job is in non-production period (runner not active)

        # Runner is active, so job is not in non-production period
        return False

    def _calculate_hourly_utilization(self, date: str, job_data: List[Dict]) -> Dict[int, float]:
        """Calculate utilization percentage for each hour of the day"""
        date_obj = datetime.date.fromisoformat(date)
        hourly_durations = defaultdict(float)

        # Create UTC day boundaries for consistent comparison
        day_start_utc = datetime.datetime.combine(date_obj, datetime.time.min, tzinfo=datetime.timezone.utc)
        day_end_utc = datetime.datetime.combine(date_obj, datetime.time.max, tzinfo=datetime.timezone.utc)

        # Get production runners for this date
        production_runners = self._get_production_runners(date, job_data)
        num_production_runners = max(1, len(production_runners))

        for job in job_data:
            if job is None:
                continue
            try:
                started_at = datetime.datetime.fromisoformat(job["started_at"])
                finished_at = datetime.datetime.fromisoformat(job["finished_at"])

                # Convert to UTC for consistent processing
                started_at_utc = started_at.astimezone(datetime.timezone.utc)
                finished_at_utc = finished_at.astimezone(datetime.timezone.utc)

                # Get the UTC date of the job
                job_date_utc = started_at_utc.date()

                # Check if job is on the target date (assuming target date is in UTC)
                if job_date_utc != date_obj:
                    continue

                # Check if job overlaps with the current day (all in UTC)
                if started_at_utc < day_end_utc and finished_at_utc > day_start_utc:
                    # Allocate job duration across hours (using UTC times)
                    current_time_utc = started_at_utc.replace(minute=0, second=0, microsecond=0)
                    end_time_utc = finished_at_utc.replace(minute=0, second=0, microsecond=0)

                    while current_time_utc <= end_time_utc:
                        hour = current_time_utc.hour  # UTC hour

                        # Calculate how much of the job duration falls in this hour
                        hour_start_utc = current_time_utc
                        hour_end_utc = current_time_utc + datetime.timedelta(hours=1)

                        job_start_in_hour = max(started_at_utc, hour_start_utc)
                        job_end_in_hour = min(finished_at_utc, hour_end_utc)

                        if job_start_in_hour < job_end_in_hour:
                            duration_in_hour = (job_end_in_hour - job_start_in_hour).total_seconds()
                            hourly_durations[hour] += duration_in_hour

                        current_time_utc += datetime.timedelta(hours=1)

            except (ValueError, TypeError, KeyError, AttributeError):
                continue

        # Convert durations to utilization percentages
        hourly_utilization = {}
        hour_capacity = num_production_runners * 3600  # seconds in an hour
        for hour, total_duration in hourly_durations.items():
            hourly_utilization[hour] = 100 * total_duration / hour_capacity

        return hourly_utilization

    def _get_production_runners(self, date: str, job_data: List[Dict]) -> set[str]:
        """Get set of runners that were in production on a given date"""
        date_obj = datetime.date.fromisoformat(date)

        # Get all unique runners that appear in the job data
        all_runners = set()
        for job in job_data:
            if job is None:
                continue
            try:
                runner_id = self._get_runner_id(job)
                all_runners.add(runner_id)
            except (ValueError, TypeError, KeyError, AttributeError):
                continue

        # Filter to only include runners that were in production on this date
        production_runners = set()
        for runner_id in all_runners:
            if self.config.is_runner_active(runner_id, date_obj):
                production_runners.add(runner_id)

        return production_runners

    def _get_runner_id(self, job: Dict) -> str:
        """Get runner identifier from job data (processed format from ci_visualization.py)"""
        runner = job.get("runner")
        if runner is None or runner == "":
            return "unknown-no-id"
        return runner

    def _filter_jobs_by_date_range(
        self, jobs: List[Dict], start_date: datetime.date, end_date: datetime.date
    ) -> List[Dict]:
        """Filter jobs to a specific date range"""
        filtered_jobs = []
        for job in jobs:
            try:
                started_at = job["started_at"]
                if started_at.endswith("Z"):
                    started_at = started_at[:-1] + "+00:00"
                started_at_dt = datetime.datetime.fromisoformat(started_at)
                job_date = started_at_dt.astimezone(datetime.timezone.utc).date()

                if start_date <= job_date <= end_date:
                    filtered_jobs.append(job)
            except (ValueError, TypeError):
                continue

        return filtered_jobs

    def _get_jobs_for_date(self, jobs: List[Dict], target_date: datetime.date) -> List[Dict]:
        """Get all jobs that occurred on a specific date"""
        date_jobs = []
        for job in jobs:
            try:
                started_at = job["started_at"]
                if started_at.endswith("Z"):
                    started_at = started_at[:-1] + "+00:00"
                started_at_dt = datetime.datetime.fromisoformat(started_at)
                job_date = started_at_dt.astimezone(datetime.timezone.utc).date()

                if job_date == target_date:
                    date_jobs.append(job)
            except (ValueError, TypeError):
                continue

        return date_jobs

    def _calculate_daily_queue_time(self, date_jobs: List[Dict]) -> float:
        """Calculate average queue time for build+test jobs on a specific date"""
        queue_times = []
        for job in date_jobs:
            if job is None:
                continue
            try:
                job_name = job.get("name", "")
                queued_duration = job.get("queued_duration")
                if job_name == "build+test" and queued_duration is not None:
                    queue_times.append(queued_duration)
            except (ValueError, TypeError, KeyError):
                continue

        return sum(queue_times) / len(queue_times) if queue_times else 0.0

    def _compute_daily_metrics_for_range(
        self, jobs: List[Dict], start_date: datetime.date, end_date: datetime.date
    ) -> Dict[str, Dict]:
        """Compute daily metrics for each day in the given range"""
        daily_metrics = {}
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.isoformat()
            date_jobs = self._get_jobs_for_date(jobs, current_date)

            # Get production runners for this date
            production_runners = self._get_production_runners(date_str, jobs)
            production_runner_count = len(production_runners)

            # Compute metrics for this date
            hourly_utilization = self._calculate_hourly_utilization(date_str, date_jobs)
            daily_utilization = (
                sum(hourly_utilization.values()) / len(hourly_utilization) if hourly_utilization else 0.0
            )

            peak_hour = 0
            peak_utilization = 0.0
            if hourly_utilization:
                peak_hour, peak_utilization = max(hourly_utilization.items(), key=lambda x: x[1])

            # Calculate average queue time for build+test jobs on this date
            daily_queue_time = self._calculate_daily_queue_time(date_jobs)

            daily_metrics[date_str] = {
                "daily_average": daily_utilization,
                "peak_hour": peak_hour,
                "peak_hour_average": peak_utilization,
                "hourly_utilization": hourly_utilization,
                "production_runner_count": production_runner_count,
                "daily_queue_time": daily_queue_time,
            }

            current_date += datetime.timedelta(days=1)

        return daily_metrics

    def _calculate_last_n_days_averages(
        self, daily_metrics: Dict[str, Dict], end_date: datetime.date, days: int
    ) -> Dict[str, float]:
        """Calculate average utilization and queue time metrics for the last N days"""
        start_date = end_date - datetime.timedelta(days=days)
        recent_metrics = {k: v for k, v in daily_metrics.items() if datetime.date.fromisoformat(k) >= start_date}

        if recent_metrics:
            avg_utilization = sum(v["daily_average"] for v in recent_metrics.values()) / len(recent_metrics)
            avg_peak_utilization = sum(v["peak_hour_average"] for v in recent_metrics.values()) / len(recent_metrics)
            avg_queue_time = sum(v["daily_queue_time"] for v in recent_metrics.values()) / len(recent_metrics)
        else:
            avg_utilization = 0.0
            avg_peak_utilization = 0.0
            avg_queue_time = 0.0

        return {
            "average_utilization": avg_utilization,
            "average_peak_utilization": avg_peak_utilization,
            "average_queue_time": avg_queue_time,
        }

    def _compute_hourly_trends(
        self, daily_metrics: Dict[str, Dict], end_date: datetime.date, days: int, jobs: List[Dict]
    ) -> Dict[str, Dict]:
        """Compute hourly trends for the last N days"""
        start_date = end_date - datetime.timedelta(days=days)
        recent_metrics = {k: v for k, v in daily_metrics.items() if datetime.date.fromisoformat(k) >= start_date}

        hourly_trends = {}
        hourly_queue_times = {}

        # Collect hourly utilization
        for date_str, metrics in recent_metrics.items():
            date_obj = datetime.date.fromisoformat(date_str)
            for hour, utilization_pct in metrics["hourly_utilization"].items():
                hour_datetime = datetime.datetime.combine(
                    date_obj, datetime.time(hour=hour), tzinfo=datetime.timezone.utc
                )
                hour_key = hour_datetime.isoformat()
                hourly_trends[hour_key] = utilization_pct

        # Calculate hourly queue times for the same period
        for current_date in [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]:
            date_jobs = self._get_jobs_for_date(jobs, current_date)
            for job in date_jobs:
                if job is None:
                    continue
                try:
                    job_name = job.get("name", "")
                    queued_duration = job.get("queued_duration")
                    if job_name == "build+test" and queued_duration is not None:
                        started_at = datetime.datetime.fromisoformat(job["started_at"])
                        started_at_utc = started_at.astimezone(datetime.timezone.utc)

                        # Create hour key for this job
                        hour_datetime = started_at_utc.replace(minute=0, second=0, microsecond=0)
                        hour_key = hour_datetime.isoformat()

                        if hour_key not in hourly_queue_times:
                            hourly_queue_times[hour_key] = []
                        hourly_queue_times[hour_key].append(queued_duration)
                except (ValueError, TypeError, KeyError):
                    continue

        # Calculate average queue times for each hour
        avg_hourly_queue_times = {}
        for hour_key, queue_times in hourly_queue_times.items():
            if queue_times:
                avg_hourly_queue_times[hour_key] = sum(queue_times) / len(queue_times)

        return {"utilization": hourly_trends, "queue_times": avg_hourly_queue_times}

    def _calculate_moving_average(self, values: List[float], window_size: int) -> List[float]:
        """Calculate moving average for a series of values"""
        moving_average = []
        for i in range(len(values)):
            if i < window_size - 1:
                window_data = values[: i + 1]
            else:
                window_data = values[i - window_size + 1 : i + 1]

            if window_data:
                avg = sum(window_data) / len(window_data)
                moving_average.append(avg)
            else:
                moving_average.append(0)

        return moving_average

    def compute_comprehensive_metrics(self, job_data: List[Dict]) -> Dict[str, Any]:
        """Compute all capacity metrics once for the past year and store results"""

        # Filter production jobs and set up date range
        filtered_jobs = self.filter_production_jobs(job_data)
        # Stop at yesterday to avoid incomplete data for today
        end_date = datetime.datetime.now().date() - datetime.timedelta(days=1)
        start_date = end_date - datetime.timedelta(days=365)

        # Filter jobs to past year
        year_jobs = self._filter_jobs_by_date_range(filtered_jobs, start_date, end_date)

        # Get unique runners
        runners = list(set(self._get_runner_id(job) for job in year_jobs))

        # Compute daily metrics for all days in the year
        daily_metrics = self._compute_daily_metrics_for_range(year_jobs, start_date, end_date)

        # Calculate averages for last 15 days
        last_15_days_averages = self._calculate_last_n_days_averages(daily_metrics, end_date, 15)

        # Compute hourly trends for last 15 days
        hourly_trends = self._compute_hourly_trends(daily_metrics, end_date, 15, year_jobs)

        # Calculate 7-day moving average for peak utilization
        sorted_dates = sorted(daily_metrics.keys())
        daily_peak_values = [daily_metrics[date]["peak_hour_average"] for date in sorted_dates]
        peak_moving_average = self._calculate_moving_average(daily_peak_values, window_size=7)

        # Store comprehensive results
        self.computed_metrics = {
            "all_jobs": year_jobs,
            "filtered_jobs": year_jobs,  # Already filtered
            "runners": runners,
            "daily_metrics": daily_metrics,
            "last_15_days": last_15_days_averages,
            "trends": {"daily": daily_metrics, "hourly": hourly_trends},
            "peak_moving_average": peak_moving_average,
            "sorted_dates": sorted_dates,
        }

        return self.computed_metrics


# HTML Generation Functions


def _build_chart_data_from_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Build the chart-data dict (dailyData, hourlyData, currentMetrics) from compute_comprehensive_metrics result."""
    trends = metrics["trends"]
    current_metrics = metrics["last_15_days"]
    daily_data = trends.get("daily", {})
    hourly_data = trends.get("hourly", {})
    hourly_queue_data = hourly_data.get("queue_times", {}) if isinstance(hourly_data, dict) else {}
    hourly_utilization_data = hourly_data.get("utilization", {}) if isinstance(hourly_data, dict) else hourly_data
    sorted_dates = metrics["sorted_dates"]
    peak_moving_average = metrics["peak_moving_average"]

    daily_utilization_values = [daily_data[date]["daily_average"] for date in sorted_dates]
    daily_peak_values = [daily_data[date]["peak_hour_average"] for date in sorted_dates]
    daily_runner_counts = [daily_data[date]["production_runner_count"] for date in sorted_dates]
    daily_queue_times = [daily_data[date]["daily_queue_time"] for date in sorted_dates]

    sorted_hourly_keys = sorted(hourly_utilization_data.keys())
    hourly_labels = []
    hourly_utilization_values = []
    hourly_queue_values = []
    for timestamp_key in sorted_hourly_keys:
        try:
            dt = datetime.datetime.fromisoformat(timestamp_key)
            hourly_labels.append(dt.strftime("%m/%d %H:%M UTC"))
            hourly_utilization_values.append(hourly_utilization_data[timestamp_key])
            hourly_queue_values.append(hourly_queue_data.get(timestamp_key, 0))
        except ValueError:
            continue

    return {
        "sorted_dates": sorted_dates,
        "daily_utilization_values": daily_utilization_values,
        "daily_peak_values": daily_peak_values,
        "daily_runner_counts": daily_runner_counts,
        "daily_queue_times": daily_queue_times,
        "peak_moving_average": peak_moving_average,
        "hourly_labels": hourly_labels,
        "hourly_utilization_values": hourly_utilization_values,
        "hourly_queue_values": hourly_queue_values,
        "current_metrics": current_metrics,
    }


def generate_capacity_page(job_data: List[Dict], output_dir: str) -> str:
    """Generate single capacity analysis HTML page automatically"""
    analyzer = CapacityAnalyzer()  # Auto-loads ci_runner_config.json

    # Compute metrics for all jobs (NRS + non-NRS)
    metrics_full = analyzer.compute_comprehensive_metrics(job_data)
    runners = metrics_full["runners"]
    filtered_jobs = metrics_full["filtered_jobs"]

    # If there are non-NRS jobs, also compute NRS-only metrics for the toggle
    nrs_only_jobs = [j for j in job_data if not j.get("is_non_nrs_project")]
    has_non_nrs = len(nrs_only_jobs) < len(job_data)
    metrics_nrs = None
    if has_non_nrs and nrs_only_jobs:
        analyzer_nrs = CapacityAnalyzer()
        metrics_nrs = analyzer_nrs.compute_comprehensive_metrics(nrs_only_jobs)

    # Generate HTML content
    html_content = _generate_capacity_html_content(
        metrics_full["trends"],
        metrics_full["last_15_days"],
        runners,
        filtered_jobs,
        metrics_full,
        metrics_nrs=metrics_nrs,
    )

    # Write to file
    capacity_path = os.path.join(output_dir, "capacity.html")
    with open(capacity_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return capacity_path


def _generate_capacity_html_content(
    trends: Dict[str, Any],
    current_metrics: Dict[str, float],
    runners: List[str],
    job_data: List[Dict],
    precomputed_metrics: Dict[str, Any],
    metrics_nrs: Dict[str, Any] | None = None,
) -> str:
    """Generate the HTML content for capacity analysis page"""

    # Build chart data for full (and optionally NRS-only when non-NRS jobs exist)
    chart_data_full = _build_chart_data_from_metrics(precomputed_metrics)
    has_non_nrs_toggle = metrics_nrs is not None
    chart_data_nrs = _build_chart_data_from_metrics(metrics_nrs) if metrics_nrs else None

    # Find the last job and get production runners at that time
    last_job = None
    if job_data:
        # Sort jobs by started_at to find the most recent
        sorted_jobs = sorted(job_data, key=lambda j: j.get("started_at", ""), reverse=True)
        last_job = sorted_jobs[0] if sorted_jobs else None

    # Get production runners at the time of the last job
    production_runners = []
    if last_job:
        try:
            # Parse the last job date
            started_at = last_job["started_at"]
            if started_at.endswith("Z"):
                started_at = started_at[:-1] + "+00:00"
            last_job_date = datetime.datetime.fromisoformat(started_at).astimezone(datetime.timezone.utc).date()

            # Create a temporary analyzer to get production runners
            analyzer = CapacityAnalyzer()
            analyzer.config.compute_production_periods(job_data)
            production_runners_set = analyzer._get_production_runners(last_job_date.isoformat(), job_data)
            production_runners = sorted(list(production_runners_set))
        except Exception:
            production_runners = sorted(runners)  # Fallback to all runners
    else:
        production_runners = sorted(runners)  # Fallback if no jobs

    d = chart_data_full
    sorted_dates = d["sorted_dates"]
    daily_utilization_values = d["daily_utilization_values"]
    daily_peak_values = d["daily_peak_values"]
    daily_runner_counts = d["daily_runner_counts"]
    daily_queue_times = d["daily_queue_times"]
    peak_moving_average = d["peak_moving_average"]
    hourly_labels = d["hourly_labels"]
    hourly_utilization_values = d["hourly_utilization_values"]
    hourly_queue_values = d["hourly_queue_values"]

    # Optional: NRS-only chart data for toggle (when non-NRS jobs exist)
    non_nrs_toggle_html = ""
    non_nrs_toggle_script = ""
    if has_non_nrs_toggle and chart_data_nrs:
        non_nrs_toggle_html = """
        <div class="non-nrs-toggle">
            <label class="toggle-switch">
                <input type="checkbox" id="hide-non-nrs-toggle">
                <span class="slider"></span>
            </label>
            <span class="filter-label">Hide contribution from non-NRS jobs</span>
        </div>"""
        nrs_data_json = json.dumps(chart_data_nrs)
        full_data_json = json.dumps(chart_data_full)
        non_nrs_toggle_script = f"""
        const fullChartData = {full_data_json};
        const nrsOnlyChartData = {nrs_data_json};
        document.getElementById('hide-non-nrs-toggle').addEventListener('change', function() {{
            const useNrsOnly = this.checked;
            const data = useNrsOnly ? nrsOnlyChartData : fullChartData;
            updateCapacityCharts(data);
            document.getElementById('avg-queue-time-value').textContent = formatDuration(data.current_metrics.average_queue_time || 0);
            document.getElementById('avg-utilization-value').textContent = data.current_metrics.average_utilization.toFixed(1) + '%';
            document.getElementById('avg-peak-utilization-value').textContent = data.current_metrics.average_peak_utilization.toFixed(1) + '%';
        }});
        function updateCapacityCharts(data) {{
            if (typeof dailyChart !== 'undefined' && dailyChart) {{
                dailyChart.data.labels = data.sorted_dates;
                dailyChart.data.datasets[0].data = data.daily_utilization_values;
                dailyChart.data.datasets[1].data = data.daily_peak_values;
                dailyChart.data.datasets[2].data = data.peak_moving_average;
                dailyChart.data.datasets[4].data = data.daily_queue_times;
                dailyChart.update('none');
            }}
            if (typeof hourlyChart !== 'undefined' && hourlyChart) {{
                hourlyChart.data.labels = data.hourly_labels;
                hourlyChart.data.datasets[0].data = data.hourly_utilization_values;
                hourlyChart.data.datasets[1].data = Array(data.hourly_labels.length).fill(70);
                hourlyChart.data.datasets[2].data = data.hourly_queue_values;
                hourlyChart.update('none');
            }}
            if (typeof runnerCountChart !== 'undefined' && runnerCountChart) {{
                runnerCountChart.data.labels = data.sorted_dates;
                runnerCountChart.data.datasets[0].data = data.daily_runner_counts;
                runnerCountChart.update('none');
            }}
        }}"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Capacity Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            width: 95%;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
            margin-top: 10px;
            margin-bottom: 10px;
        }}
        .chart-container {{
            margin: 20px 0;
            height: 400px;
            position: relative;
        }}
        .back-link {{
            display: inline-block;
            margin-bottom: 15px;
            padding: 5px 10px;
            background-color: #f0f0f0;
            border-radius: 3px;
            text-decoration: none;
            color: #333;
            font-size: 14px;
        }}
        .back-link:hover {{
            background-color: #e0e0e0;
        }}
        .utilization-metrics {{
            display: flex;
            gap: 30px;
            margin: 15px 0;
            flex-wrap: wrap;
        }}
        .metric {{
            display: flex;
            flex-direction: column;
            align-items: center;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #dee2e6;
            min-width: 200px;
        }}
        .metric-label {{
            font-size: 14px;
            color: #6c757d;
            margin-bottom: 5px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .runner-info {{
            margin: 10px 0;
            padding: 10px;
            background-color: #e9ecef;
            border-radius: 3px;
        }}
        .production-runners {{
            margin-top: 15px;
        }}
        .production-runners h4 {{
            margin-bottom: 10px;
            color: #495057;
        }}
        .production-runners ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .production-runners li {{
            margin: 5px 0;
            font-family: monospace;
            background-color: #f8f9fa;
            padding: 2px 5px;
            border-radius: 3px;
        }}
        .toggle-switch {{
            position: relative;
            display: inline-block;
            width: 50px;
            height: 28px;
            margin-right: 8px;
        }}
        .toggle-switch input {{
            opacity: 0;
            width: 0;
            height: 0;
        }}
        .toggle-switch .slider {{
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 28px;
        }}
        .toggle-switch .slider:before {{
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }}
        .toggle-switch input:checked + .slider {{
            background-color: #2196F3;
        }}
        .toggle-switch input:checked + .slider:before {{
            transform: translateX(22px);
        }}
        .non-nrs-toggle {{
            display: flex;
            align-items: center;
            margin: 15px 0;
        }}
        .non-nrs-toggle .filter-label {{
            margin-left: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="index.html" class="back-link">← Back to Index</a>
        <h1>CI Capacity Analysis</h1>
        {non_nrs_toggle_html}

        <div class="runner-info">
            <h3>Current System Status (Last 15 Days)</h3>
            <div class="utilization-metrics">
                <div class="metric">
                    <div class="metric-label">Average Utilization:</div>
                    <div class="metric-value" id="avg-utilization-value">{current_metrics["average_utilization"]:.1f}%</div>
            </div>
                <div class="metric">
                    <div class="metric-label">Average Peak Hour Utilization:</div>
                    <div class="metric-value" id="avg-peak-utilization-value">{current_metrics["average_peak_utilization"]:.1f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Average build+test Queue Time:</div>
                    <div class="metric-value" id="avg-queue-time-value">--</div>
                </div>
            </div>
            <div class="production-runners">
                <h4>Production Runners (at time of last job):</h4>
                <ul>
                    {"".join(f"<li>{runner}</li>" for runner in production_runners) if production_runners else "<li>None</li>"}
                </ul>
            </div>
        </div>

        <div class="chart-container">
            <h3>Hourly Utilization (Last 15 Days, UTC)</h3>
            <canvas id="hourlyChart"></canvas>
        </div>

        <div class="chart-container">
            <h3>Daily Utilization Trends (Last Year, UTC)</h3>
            <canvas id="dailyChart"></canvas>
        </div>

        <div class="chart-container">
            <h3>Production Runners Count (Last Year, UTC)</h3>
            <canvas id="runnerCountChart"></canvas>
        </div>

    </div>

    <script>
        // Chart data
        const dailyData = {{
            labels: {json.dumps(sorted_dates)},
            dailyUtilization: {json.dumps(daily_utilization_values)},
            peakUtilization: {json.dumps(daily_peak_values)},
            peakMovingAverage: {json.dumps(peak_moving_average)},
            runnerCounts: {json.dumps(daily_runner_counts)},
            queueTimes: {json.dumps(daily_queue_times)}
        }};

        const hourlyData = {{
            labels: {json.dumps(hourly_labels)},
            utilization: {json.dumps(hourly_utilization_values)},
            queueTimes: {json.dumps(hourly_queue_values)}
        }};


        const currentUtilization = {current_metrics["average_utilization"]};

        // Helper function to format queue time duration
        function formatDuration(seconds) {{
            if (seconds === 0 || seconds === null || seconds === undefined) return '0s';
            
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = Math.round(seconds % 60);
            
            if (minutes === 0) return `${{remainingSeconds}}s`;
            if (minutes < 60) return `${{minutes}}m ${{remainingSeconds}}s`;
            
            const hours = Math.floor(minutes / 60);
            const remainingMinutes = minutes % 60;
            return `${{hours}}h ${{remainingMinutes}}m`;
        }}

        // Initialize charts
        let dailyChart, hourlyChart;

        document.addEventListener('DOMContentLoaded', function() {{
            initializeCharts();
        }});

        function initializeCharts() {{
            // Daily utilization chart
            const dailyCtx = document.getElementById('dailyChart').getContext('2d');
            dailyChart = new Chart(dailyCtx, {{
                type: 'line',
                data: {{
                    labels: dailyData.labels,
                    datasets: [
                        {{
                            label: 'Daily Average Utilization (%)',
                            data: dailyData.dailyUtilization,
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                            tension: 0.4,
                            fill: true,
                            yAxisID: 'y'
                        }},
                        {{
                            label: 'Peak Hour Average Utilization (%)',
                            data: dailyData.peakUtilization,
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.1)',
                            tension: 0.4,
                            fill: false,
                            yAxisID: 'y'
                        }},
                        {{
                            label: 'Peak Hour 7-Day Moving Average (%)',
                            data: dailyData.peakMovingAverage,
                            borderColor: 'rgba(156, 39, 176, 1)',
                            backgroundColor: 'rgba(156, 39, 176, 0.1)',
                            borderWidth: 3,
                            tension: 0.4,
                            fill: false,
                            pointRadius: 0,
                            yAxisID: 'y'
                        }},
                        {{
                            label: '70% Target',
                            data: Array(dailyData.labels.length).fill(70),
                            borderColor: 'rgba(255, 193, 7, 1)',
                            backgroundColor: 'rgba(255, 193, 7, 0.1)',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            pointRadius: 0,
                            fill: false,
                            yAxisID: 'y'
                        }},
                        {{
                            label: 'build+test Queue Time',
                            data: dailyData.queueTimes,
                            type: 'bar',
                            backgroundColor: 'rgba(255, 159, 64, 0.3)',
                            borderColor: 'rgba(255, 159, 64, 1)',
                            borderWidth: 1,
                            yAxisID: 'y1'
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{
                        mode: 'index',
                        intersect: false
                    }},
                    scales: {{
                        y: {{
                            type: 'linear',
                            display: true,
                            position: 'left',
                            beginAtZero: true,
                            max: 100,
                            title: {{
                                display: true,
                                text: 'Utilization (%)'
                            }}
                        }},
                        y1: {{
                            type: 'linear',
                            display: true,
                            position: 'right',
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Queue Time'
                            }},
                            grid: {{
                                drawOnChartArea: false
                            }},
                            ticks: {{
                                callback: function(value) {{
                                    return formatDuration(value);
                                }}
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Date (UTC)'
                            }}
                        }}
                    }},
                    plugins: {{
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    if (context.dataset.label.includes('Queue Time')) {{
                                        return `${{context.dataset.label}}: ${{formatDuration(context.raw)}}`;
                                    }}
                                    return `${{context.dataset.label}}: ${{context.formattedValue}}${{context.dataset.label.includes('%') ? '' : '%'}}`;
                                }}
                            }}
                        }},
                        legend: {{
                            display: true,
                            position: 'top'
                        }}
                    }}
                }}
            }});

            // Hourly utilization chart
            const hourlyCtx = document.getElementById('hourlyChart').getContext('2d');
            hourlyChart = new Chart(hourlyCtx, {{
                type: 'line',
                data: {{
                    labels: hourlyData.labels,
                    datasets: [
                        {{
                            label: 'Utilization (%)',
                            data: hourlyData.utilization,
                            backgroundColor: 'rgba(75, 192, 192, 0.1)',
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 2,
                            tension: 0.4,
                            fill: true,
                            yAxisID: 'y'
                        }},
                        {{
                            label: '70% Target',
                            data: Array(hourlyData.labels.length).fill(70),
                            borderColor: 'rgba(255, 193, 7, 1)',
                            backgroundColor: 'rgba(255, 193, 7, 0.1)',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            pointRadius: 0,
                            fill: false,
                            yAxisID: 'y'
                        }},
                        {{
                            label: 'build+test Queue Time',
                            data: hourlyData.queueTimes,
                            type: 'bar',
                            backgroundColor: 'rgba(255, 159, 64, 0.3)',
                            borderColor: 'rgba(255, 159, 64, 1)',
                            borderWidth: 1,
                            yAxisID: 'y1'
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{
                        mode: 'index',
                        intersect: false
                    }},
                    scales: {{
                        y: {{
                            type: 'linear',
                            display: true,
                            position: 'left',
                            beginAtZero: true,
                            max: 100,
                            title: {{
                                display: true,
                                text: 'Utilization (%)'
                            }}
                        }},
                        y1: {{
                            type: 'linear',
                            display: true,
                            position: 'right',
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Queue Time'
                            }},
                            grid: {{
                                drawOnChartArea: false
                            }},
                            ticks: {{
                                callback: function(value) {{
                                    return formatDuration(value);
                                }}
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Time (UTC)'
                            }}
                        }}
                    }},
                    plugins: {{
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    if (context.dataset.label.includes('Queue Time')) {{
                                        return `${{context.dataset.label}}: ${{formatDuration(context.raw)}}`;
                                    }}
                                    return `${{context.dataset.label}}: ${{context.formattedValue}}${{context.dataset.label.includes('%') ? '' : '%'}}`;
                                }}
                            }}
                        }}
                    }}
                }}
            }});

            // Production runners count chart
            const runnerCountCtx = document.getElementById('runnerCountChart').getContext('2d');
            runnerCountChart = new Chart(runnerCountCtx, {{
                type: 'line',
                data: {{
                    labels: dailyData.labels,
                    datasets: [
                        {{
                            label: 'Production Runners',
                            data: dailyData.runnerCounts,
                            backgroundColor: 'rgba(153, 102, 255, 0.1)',
                            borderColor: 'rgba(153, 102, 255, 1)',
                            borderWidth: 2,
                            tension: 0,
                            fill: true,
                            pointRadius: 0,
                            pointHoverRadius: 4
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Number of Runners'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Date (UTC)'
                            }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            display: true
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false
                        }}
                    }}
                }}
            }});

            // Format the average queue time display
            const avgQueueTimeElement = document.getElementById('avg-queue-time-value');
            if (avgQueueTimeElement) {{
                const queueTimeSeconds = {current_metrics.get("average_queue_time", 0)};
                avgQueueTimeElement.textContent = formatDuration(queueTimeSeconds);
            }}
            {non_nrs_toggle_script}

        }}

    </script>
</body>
</html>"""
