"""
Tests for plotting utility functions.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from pathlib import Path
from plotting.utils import md_pattern_like, plot_missing_data_pattern


def _heatmap(fig):
    """The single axes of a pattern plot and its cell colours, one row per pattern."""
    (ax,) = fig.axes
    mesh = ax.collections[0]
    return ax, mesh.get_array().reshape(mesh.get_coordinates().shape[0] - 1, -1)


class TestMdPatternLike:
    """Test missing data pattern analysis."""
    
    def test_md_pattern_basic(self, sample_data_missing):
        """Test basic missing data pattern analysis."""
        result = md_pattern_like(sample_data_missing)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == len(sample_data_missing.columns) + 1
        pattern_cols = [col for col in result.columns if col != '#miss_row']
        pattern_data = result.iloc[:-1][pattern_cols]
        assert pattern_data.isin([0, 1]).all().all()
        
    def test_md_pattern_complete_data(self, sample_data_complete):
        """Test missing data pattern with complete data."""
        result = md_pattern_like(sample_data_complete)

        # Should have 2 rows: 1 pattern row + 1 #miss_col row
        assert len(result) == 2
        # Check that the pattern row (first row) has all 1s for data columns
        pattern_cols = [col for col in result.columns if col != '#miss_row']
        pattern_row = result.iloc[0][pattern_cols]
        assert (pattern_row == 1).all()
        
    def test_md_pattern_no_data(self):
        """Test missing data pattern with empty dataframe."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Data must have at least two columns"):
            md_pattern_like(empty_df)
        
    def test_md_pattern_all_missing(self):
        """Test missing data pattern with all missing data."""
        all_missing = pd.DataFrame({
            'A': [np.nan, np.nan, np.nan],
            'B': [np.nan, np.nan, np.nan]
        })
        result = md_pattern_like(all_missing)
        
        assert len(result) == 2
        pattern_cols = [col for col in result.columns if col != '#miss_row']
        pattern_row = result.iloc[0][pattern_cols]
        assert (pattern_row == 0).all()
        
    def test_md_pattern_mixed_missing(self):
        """Test missing data pattern with mixed missing patterns."""
        mixed_data = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan],
            'B': [1, 2, np.nan, np.nan],
            'C': [1, 2, 3, 4]
        })
        result = md_pattern_like(mixed_data)

        assert len(result) > 1
        assert '#miss_row' in result.columns
        assert len(result.columns) == 4

    def test_md_pattern_row_order_is_r_order(self):
        """Patterns sorted as R's md.pattern sorts them, not by number of missing cells.

        R 4.6.2, mice 3.19.0, md.pattern(data.frame(A = c(1, NA, 3, NA),
        B = c(1, 2, NA, NA), C = 1:4), plot = FALSE):
              C A B
            1 1 1 1 0
            1 1 1 0 1
            1 1 0 1 1
            1 1 0 0 2
              0 2 2 4
        """
        mixed_data = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan],
            'B': [1, 2, np.nan, np.nan],
            'C': [1, 2, 3, 4]
        })
        result = md_pattern_like(mixed_data)

        assert list(result.columns) == ['C', 'A', 'B', '#miss_row']
        assert list(result.index) == ['1', '1', '1', '1', '#miss_col']
        np.testing.assert_array_equal(
            result.to_numpy(),
            [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 2], [0, 2, 2, 4]],
        )


class TestPlotMissingDataPattern:
    """Test missing data pattern visualization."""
    
    def test_plot_missing_pattern_basic(self, sample_data_missing):
        """Test basic missing data pattern plotting."""
        pattern_df = md_pattern_like(sample_data_missing)
        
        fig = plot_missing_data_pattern(pattern_df)
        assert isinstance(fig, Figure)
        ax, cells = _heatmap(fig)
        # One cell per pattern and column; 1 marks a missing cell
        n_patterns, n_columns = pattern_df.shape[0] - 1, pattern_df.shape[1] - 1
        assert cells.shape == (n_patterns, n_columns)
        np.testing.assert_array_equal(cells, 1 - pattern_df.iloc[:-1, :-1].to_numpy())
        texts = [t.get_text() for t in ax.texts]
        assert texts[:n_patterns] == list(pattern_df.index[:-1])
        assert ax.get_title() == "Missing Data Pattern"
        plt.close('all')
        
    def test_plot_missing_pattern_save(self, sample_data_missing, temp_dir):
        """Test missing data pattern plotting with save."""
        pattern_df = md_pattern_like(sample_data_missing)
        save_path = temp_dir / "missing_pattern_test.png"
        
        fig = plot_missing_data_pattern(
            pattern_df,
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        assert isinstance(fig, Figure)
        assert fig.number not in plt.get_fignums()
        plt.close('all')
        
    def test_plot_missing_pattern_custom_title(self, sample_data_missing):
        """Test missing data pattern plotting with custom title."""
        pattern_df = md_pattern_like(sample_data_missing)
        
        fig = plot_missing_data_pattern(
            pattern_df,
            title="Custom Missing Data Pattern"
        )
        assert fig.axes[0].get_title() == "Custom Missing Data Pattern"
        plt.close('all')
        
    def test_plot_missing_pattern_rotated_names(self, sample_data_missing):
        """Test missing data pattern plotting with rotated names."""
        pattern_df = md_pattern_like(sample_data_missing)
        
        fig = plot_missing_data_pattern(
            pattern_df,
            rotate_names=True
        )
        names = [t for t in fig.axes[0].texts if t.get_text() in pattern_df.columns[:-1]]
        assert len(names) == pattern_df.shape[1] - 1
        assert all(t.get_rotation() == 90 for t in names)
        # Extra height for the rotated names
        np.testing.assert_allclose(fig.get_size_inches(), (8, 5.5))
        plt.close('all')
        
    def test_plot_missing_pattern_custom_figsize(self, sample_data_missing):
        """Test missing data pattern plotting with custom figure size."""
        pattern_df = md_pattern_like(sample_data_missing)
        
        fig = plot_missing_data_pattern(
            pattern_df,
            figsize=(12, 8)
        )
        np.testing.assert_allclose(fig.get_size_inches(), (12, 8))
        plt.close('all')

    @pytest.mark.parametrize("value, colour", [(np.nan, (205, 100, 140)), (1.0, (101, 155, 213))])
    def test_plot_missing_pattern_single_pattern_colour(self, value, colour):
        """All cells missing are drawn in the missing colour, all observed in the observed colour."""
        data = pd.DataFrame({'A': [value] * 3, 'B': [value] * 3})
        fig = plot_missing_data_pattern(md_pattern_like(data))
        (ax,) = fig.axes
        expected = to_rgba(tuple(c / 255 for c in colour))
        for face in ax.collections[0].get_facecolors():
            np.testing.assert_allclose(face, expected)
        plt.close('all')
    
            
    def test_plot_missing_pattern_single_column(self):
        """Test missing data pattern plotting with single column."""
        single_col_data = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan]
        })
        # Should raise ValueError for single column data
        with pytest.raises(ValueError, match="Data must have at least two columns"):
            md_pattern_like(single_col_data)


class TestUtilsIntegration:
    """Test integration between utility functions."""
    
    def test_pattern_to_plot_workflow(self, sample_data_missing, temp_dir):
        """Test complete workflow from data to pattern to plot."""
        # Step 1: Analyze missing pattern
        pattern_df = md_pattern_like(sample_data_missing)
        
        # Step 2: Plot the pattern
        save_path = temp_dir / "workflow_test.png"
        plot_missing_data_pattern(
            pattern_df,
            title="Test Workflow",
            save_path=str(save_path)
        )
        
        assert isinstance(pattern_df, pd.DataFrame)
        assert len(pattern_df) > 0
        assert save_path.exists()
        plt.close('all')
        
    def test_pattern_consistency(self, sample_data_missing):
        """Test that pattern analysis is consistent."""
        pattern1 = md_pattern_like(sample_data_missing)
        pattern2 = md_pattern_like(sample_data_missing)
        
        pd.testing.assert_frame_equal(pattern1, pattern2)
        
    def test_pattern_with_different_data_types(self):
        """Test pattern analysis with different data types."""
        mixed_types = pd.DataFrame({
            'int_col': [1, 2, np.nan, 4],
            'float_col': [1.1, np.nan, 3.3, 4.4],
            'str_col': ['A', 'B', np.nan, 'D'],
            'bool_col': [True, False, np.nan, True]
        })
        
        pattern_df = md_pattern_like(mixed_types)
        
        assert isinstance(pattern_df, pd.DataFrame)
        assert len(pattern_df) > 0
        
        fig = plot_missing_data_pattern(pattern_df)
        assert isinstance(fig, Figure)
        _, cells = _heatmap(fig)
        assert cells.shape == (pattern_df.shape[0] - 1, 4)
        plt.close('all')
