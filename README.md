# MIC vs Distance Correlation Comparison

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1126%2Fscience.1205438-blue)](https://doi.org/10.1126/science.1205438)

**Independent reproduction study validating Simon & Tibshirani's criticism of the Maximal Information Coefficient (MIC)**

> **Key Finding**: Our results fully support Simon & Tibshirani's claim that MIC has significantly lower statistical power than Distance Correlation across all relationship types.

## Table of Contents

- [Overview](#overview)
- [Scientific Context](#scientific-context)
- [Methods](#methods)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## Overview

This repository contains an independent reproduction study comparing the statistical power of three association measures:

- **Pearson Correlation** - Linear relationships benchmark
- **Distance Correlation (dcor)** - Non-linear relationships
- **Maximal Information Coefficient (MIC)** - Universal dependency measure

The study tests the controversial claim by Simon & Tibshirani that MIC has lower power than Distance Correlation in most practical scenarios.

## Scientific Context

### The Controversy

**Reshef et al. (Science, 2011)** proposed MIC as a universal dependency detector with "equitability" - equal sensitivity to all relationship types.

**Simon & Tibshirani (Stanford Comment)** criticized MIC, stating: 
> "MIC has lower power than dcor in every case except the somewhat pathological high-frequency sine wave"

### Our Contribution

We independently reproduce and extend this comparison with:
- 7 relationship types (extended from original)
- 6 noise levels (increasing difficulty)
- 12,600 total statistical tests
- Open-source, reproducible code

## Methods

### Relationship Types Tested

| Type | Formula | Description |
|------|---------|-------------|
| Linear | `y = x + noise` | Basic linear relationship |
| Quadratic | `y = 4(x-0.5)² + noise` | Simple non-linear |
| Cubic | `y = 4(x-0.5)³ + noise` | Higher-order polynomial |
| Sine (Low Freq) | `y = sin(2πx) + noise` | Smooth periodic |
| Sine (High Freq) | `y = sin(4πx) + noise` | High-frequency periodic |
| Exponential | `y = exp(x) + noise` | Exponential growth |
| Step | `y = I(x>0.5) + noise` | Discontinuous relationship |

### Experimental Design

- **Sample size**: 100 observations per simulation
- **Noise levels**: 0.1, 0.28, 0.46, 0.64, 0.82, 1.0
- **Simulations**: 100 per scenario
- **Total tests**: 7 × 6 × 3 × 100 = 12,600

## Results

### Average Power Across All Scenarios

| Method | Average Power | Rank |
|--------|---------------|------|
| **Distance Correlation** | **0.833** | 🥇 |
| Pearson Correlation | 0.762 | 🥈 |
| MIC (Alternative) | 0.405 | 🥉 |

### Power by Relationship Type

| Relationship | DCOR | MIC | Pearson | Best |
|--------------|------|-----|---------|------|
| Linear | 0.667 | 0.167 | **0.833** | Pearson |
| Quadratic | **0.833** | 0.333 | 0.000 | **DCOR** |
| Cubic | 0.333 | 0.167 | **0.500** | Pearson |
| Sine Low Freq | **1.000** | 0.833 | 1.000 | **DCOR** |
| Sine High Freq | **1.000** | 0.667 | 1.000 | **DCOR** |
| Exponential | **1.000** | 0.333 | 1.000 | **DCOR** |
| Step | **1.000** | 0.333 | 1.000 | **DCOR** |

### Key Findings

**Simon & Tibshirani Fully Supported**:
- DCOR better than MIC in **7/7 cases** (100%)
- MIC better than DCOR in **0/7 cases** (0%)
- Even in high-frequency sine: DCOR (1.000) > MIC (0.667)

**Critical Weakness**: MIC completely fails (power = 0) at highest noise level

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Setup

```bash
# Clone repository
git clone https://github.com/AnyaAleks/MIC-vs-DCor-Comparison.git
cd mic-vs-dcor-comparison

# Create virtual environment (recommended)
python -m venv stats_env
source stats_env/bin/activate  # On Windows: stats_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
