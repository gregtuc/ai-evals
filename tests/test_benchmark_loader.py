"""Tests for benchmark_loader.py: GPQA per-category sampling."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from ai_evals.benchmark_loader import _load_gpqa_diamond


class TestGPQASampling:
    """Test that GPQA Diamond samples per subdomain, not globally."""

    def _make_mock_dataset(self, n_per_subdomain: int = 20):
        """Create a mock GPQA dataset with known subdomains."""
        rows = []
        subdomains = ["Organic Chemistry", "Molecular Biology", "Quantum Mechanics"]
        for subdomain in subdomains:
            for i in range(n_per_subdomain):
                rows.append({
                    "Question": f"Question {i} about {subdomain}?",
                    "Correct Answer": f"Correct answer {i}",
                    "Incorrect Answer 1": f"Wrong 1 for {i}",
                    "Incorrect Answer 2": f"Wrong 2 for {i}",
                    "Incorrect Answer 3": f"Wrong 3 for {i}",
                    "Subdomain": subdomain,
                })
        return rows

    @patch("datasets.load_dataset")
    def test_per_category_sampling(self, mock_load_dataset):
        """sample_per_category should sample within each subdomain."""
        mock_ds = self._make_mock_dataset(n_per_subdomain=20)
        mock_load_dataset.return_value = mock_ds

        tasks = _load_gpqa_diamond(sample_per_category=5, seed=42)

        # Should have 5 per subdomain = 15 total
        categories = set(t.category for t in tasks)
        assert len(categories) == 3

        for cat in categories:
            cat_tasks = [t for t in tasks if t.category == cat]
            assert len(cat_tasks) == 5, f"Expected 5 tasks for {cat}, got {len(cat_tasks)}"

    @patch("datasets.load_dataset")
    def test_no_sampling_gets_all(self, mock_load_dataset):
        """Without sample_per_category, should get all tasks."""
        mock_ds = self._make_mock_dataset(n_per_subdomain=10)
        mock_load_dataset.return_value = mock_ds

        tasks = _load_gpqa_diamond(sample_per_category=None, seed=42)
        assert len(tasks) == 30  # 10 per subdomain * 3 subdomains

    @patch("datasets.load_dataset")
    def test_category_prefixed_ids(self, mock_load_dataset):
        """Task IDs should include the category for uniqueness."""
        mock_ds = self._make_mock_dataset(n_per_subdomain=3)
        mock_load_dataset.return_value = mock_ds

        tasks = _load_gpqa_diamond(sample_per_category=None, seed=42)

        # IDs should contain the category
        for task in tasks:
            assert task.category in task.id, f"Expected category '{task.category}' in ID '{task.id}'"

    @patch("datasets.load_dataset")
    def test_subdomain_normalization(self, mock_load_dataset):
        """Subdomains like 'Organic Chemistry' should become 'organic_chemistry'."""
        mock_ds = self._make_mock_dataset(n_per_subdomain=2)
        mock_load_dataset.return_value = mock_ds

        tasks = _load_gpqa_diamond(sample_per_category=None, seed=42)
        categories = set(t.category for t in tasks)
        assert "organic_chemistry" in categories
        assert "molecular_biology" in categories
        assert "quantum_mechanics" in categories

    @patch("datasets.load_dataset")
    def test_small_category_not_oversampled(self, mock_load_dataset):
        """If a category has fewer items than sample_per_category, use all of them."""
        rows = []
        for i in range(3):
            rows.append({
                "Question": f"Q{i}",
                "Correct Answer": f"A{i}",
                "Incorrect Answer 1": "W1",
                "Incorrect Answer 2": "W2",
                "Incorrect Answer 3": "W3",
                "Subdomain": "Tiny Category",
            })
        mock_load_dataset.return_value = rows

        tasks = _load_gpqa_diamond(sample_per_category=10, seed=42)
        assert len(tasks) == 3  # only 3 available, not 10
