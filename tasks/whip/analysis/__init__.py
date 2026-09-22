"""Saved-run analysis for the whip task."""

from pathlib import Path


# Retained historical inputs; current simulation/training does not use this archive.
ARCHIVE_ROOT = Path(__file__).resolve().parents[4]/'AeroWhip_archive/20260922-cleanup'
