"""
Tests for diagnostic plotting functions.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from plotting.diagnostics import (
    stripplot,
    bwplot,
    densityplot,
    densityplot_split,
    xyplot,
    plot_chain_stats
)


class TestStripplot:
    """Test stripplot functionality."""
    
    def test_stripplot_basic(self, imputed_datasets, missing_pattern, temp_dir):
        """Test basic stripplot creation."""
        # Should not raise an exception
        stripplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            columns=['bmi'],
            merge_imputations=False
        )
        plt.close('all')
        
    def test_stripplot_merged(self, imputed_datasets, missing_pattern):
        """Test stripplot with merged imputations."""
        stripplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            columns=['bmi'],
            merge_imputations=True
        )
        plt.close('all')
        
    def test_stripplot_save(self, imputed_datasets, missing_pattern, temp_dir):
        """Test stripplot saving to file."""
        save_path = temp_dir / "stripplot_test.png"
        
        stripplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            columns=['bmi'],
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        plt.close('all')
        
    def test_stripplot_custom_colors(self, imputed_datasets, missing_pattern):
        """Test stripplot with custom colors."""
        stripplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            columns=['bmi'],
            observed_color='green',
            imputed_color='orange'
        )
        plt.close('all')
        
    def test_stripplot_no_missing_columns(self, imputed_datasets, missing_pattern):
        """Test stripplot behavior when no columns have missing values."""
        no_missing_pattern = missing_pattern.copy()
        no_missing_pattern[:] = 1  
        
        stripplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=no_missing_pattern
        )
        plt.close('all')


class TestBwplot:
    """Test box-and-whisker plot functionality."""
    
    def test_bwplot_basic(self, imputed_datasets, missing_pattern):
        """Test basic bwplot creation."""
        bwplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            columns=['bmi']
        )
        plt.close('all')
        
    def test_bwplot_save(self, imputed_datasets, missing_pattern, temp_dir):
        """Test bwplot saving to file."""
        save_path = temp_dir / "bwplot_test.png"
        
        bwplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            columns=['bmi'],
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        plt.close('all')


class TestDensityplot:
    """Test density plot functionality."""
    
    def test_densityplot_basic(self, imputed_datasets, missing_pattern):
        """Test basic densityplot creation."""
        densityplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            columns=['bmi']
        )
        plt.close('all')
        
    def test_densityplot_merged(self, imputed_datasets, missing_pattern):
        """Test densityplot with merged imputations."""
        densityplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            columns=['bmi']
        )
        plt.close('all')
        
    def test_densityplot_split(self, imputed_datasets, missing_pattern):
        """Test split density plot."""
        densityplot_split(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            column='bmi'
        )
        plt.close('all')
        
    def test_densityplot_split_save(self, imputed_datasets, missing_pattern, temp_dir):
        """Test split density plot saving."""
        save_path = temp_dir / "densityplot_split_test.png"
        
        densityplot_split(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            column='bmi',
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        plt.close('all')


class TestXyplot:
    """Test scatter plot (xyplot) functionality."""
    
    def test_xyplot_basic(self, imputed_datasets, missing_pattern):
        """Test basic xyplot creation."""
        xyplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            x='age',
            y='bmi'
        )
        plt.close('all')
        
    def test_xyplot_merged(self, imputed_datasets, missing_pattern):
        """Test xyplot with merged imputations."""
        xyplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            x='age',
            y='bmi',
            merge_imputations=True
        )
        plt.close('all')
        
    def test_xyplot_save(self, imputed_datasets, missing_pattern, temp_dir):
        """Test xyplot saving to file."""
        save_path = temp_dir / "xyplot_test.png"
        
        xyplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            x='age',
            y='bmi',
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        plt.close('all')
        
    def test_xyplot_missing_columns(self, imputed_datasets, missing_pattern):
        """Test xyplot error handling for missing columns."""
        # This should handle the error gracefully (print message and return)
        xyplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            x='nonexistent_x',
            y='nonexistent_y'
        )
        plt.close('all')


class TestPlotChainStats:
    """Test chain statistics plotting."""
    
    def test_plot_chain_stats_basic(self, temp_dir):
        """Test basic chain statistics plotting."""
        # Create mock chain statistics
        n_iter, n_chains = 10, 3
        chain_mean = {
            'bmi': np.random.randn(n_iter, n_chains),
            'cholesterol': np.random.randn(n_iter, n_chains)
        }
        chain_var = {
            'bmi': np.random.rand(n_iter, n_chains),
            'cholesterol': np.random.rand(n_iter, n_chains)
        }
        
        plot_chain_stats(
            chain_mean=chain_mean,
            chain_var=chain_var,
            columns=['bmi']
        )
        plt.close('all')
        
    def test_plot_chain_stats_save(self, temp_dir):
        """Test chain statistics plotting with save."""
        n_iter, n_chains = 10, 3
        chain_mean = {
            'bmi': np.random.randn(n_iter, n_chains)
        }
        chain_var = {
            'bmi': np.random.rand(n_iter, n_chains)
        }
        
        save_path = temp_dir / "chain_stats_test.png"
        
        plot_chain_stats(
            chain_mean=chain_mean,
            chain_var=chain_var,
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        plt.close('all')
        
    def test_plot_chain_stats_no_columns(self):
        """Test chain statistics plotting with no columns."""
        chain_mean = {}
        chain_var = {}
        
        plot_chain_stats(
            chain_mean=chain_mean,
            chain_var=chain_var
        )
        plt.close('all')
        
    def test_plot_chain_stats_missing_column(self):
        """Test chain statistics plotting with missing column data."""
        chain_mean = {'bmi': np.random.randn(10, 3)}
        chain_var = {}  

        plot_chain_stats(
            chain_mean=chain_mean,
            chain_var=chain_var,
            columns=['bmi']
        )
        plt.close('all')
