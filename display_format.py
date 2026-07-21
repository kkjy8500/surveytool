import pandas as pd
import re

from config import (
    DEFAULT_PCT_DECIMALS,
    DEFAULT_STAT_DECIMALS,
    DEFAULT_N_IN_PARENTHESES
)

# ---------------------------------------------------------
# Percentage & Statistical Formatting
# ---------------------------------------------------------

def format_pct(value, decimals=DEFAULT_PCT_DECIMALS, suffix="") -> str:
    """
    Format percentages (e.g., 32.4).
    Zeros and near-zeros are formatted to the specified decimal places.
    """
    if pd.isna(value):
        return ""
        
    return f"{float(value):.{decimals}f}{suffix}"


def format_stat(value, decimals=DEFAULT_STAT_DECIMALS, suffix="") -> str:
    """
    Format statistical metrics like Mean or Standard Deviation (e.g., 3.47).
    """
    if pd.isna(value):
        return ""

    return f"{float(value):.{decimals}f}{suffix}"

# ---------------------------------------------------------
# Count & Frequency Formatting
# ---------------------------------------------------------

def format_n(value, parentheses=DEFAULT_N_IN_PARENTHESES, weighted=False) -> str:
    """
    Format case counts (N).
    If weighted=True, values are rounded to the nearest integer.
    Example output: (500) or 500.
    """
    if pd.isna(value):
        return ""
        
    try:
        if weighted:
            num_val = int(round(float(value)))
        else:
            num_val = int(float(value))
            
        text = f"{num_val:,}"
        return f"({text})" if parentheses else text
    except Exception:
        return str(value)

# ---------------------------------------------------------
# Statistical Significance Formatting
# ---------------------------------------------------------

def format_pvalue(value) -> str:
    """
    Format p-values for significance testing.
    Values less than 0.001 are formatted as '<0.001'.
    """
    if pd.isna(value):
        return ""
    
    pval = float(value)
    if pval < 0.001:
        return "<0.001"
    return f"{pval:.3f}"


# ---------------------------------------------------------
# Backward Compatibility Wrappers
# ---------------------------------------------------------

def fmt_pct(value, decimals=DEFAULT_PCT_DECIMALS) -> str:
    """Legacy alias for format_pct."""
    return format_pct(value, decimals=decimals)


def fmt_num(value, decimals=DEFAULT_STAT_DECIMALS) -> str:
    """Legacy alias for format_stat."""
    return format_stat(value, decimals=decimals)


def format_count_text(value) -> str:
    """Legacy wrapper for format_n with parentheses."""
    return format_n(value, parentheses=True)


def format_chart_count_text(value) -> str:
    """Legacy wrapper for format_n with parentheses."""
    return format_n(value, parentheses=True)
