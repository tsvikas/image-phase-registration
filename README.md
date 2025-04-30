# Image Registration (Rotation + Translation) via Phase Correlation

This repository demonstrates how to register two images when one is a rotated and translated version of the other.

## Overview

The method uses **[Phase Correlation](https://en.wikipedia.org/wiki/Phase_correlation)**,
extended to handle rotation by analyzing the **[Fourier Transform (FFT)](https://en.wikipedia.org/wiki/Fast_Fourier_transform)**
in **[polar coordinates](https://en.wikipedia.org/wiki/Polar_coordinate_system)**.
This approach is effective even for images without image features.

## Code & Explanation

**All the code, step-by-step explanations, and examples are in the [Jupyter Notebook](notebook.ipynb)**

Please refer to the notebook for detailed algorithm steps, visualizations, and usage examples.

## Setup & Dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) for managing dependencies and the virtual environment.
Dependencies are listed in the `pyproject.toml` file.

## Usage

1. **Clone the repository:**

   ```bash
   git clone github.com/tsvikas/image-phase-registration.git
   cd image-phase-registration
   ```

1. **Run with JupyterLab:**

   ```bash
   uv run jupyter lab notebook.ipynb
   ```
