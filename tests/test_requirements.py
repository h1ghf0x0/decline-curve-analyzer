"""
Tests for Requirements and Dependencies - Ensure all required packages are installed.
"""
import sys
import subprocess
from pathlib import Path

class TestRequirements:
    def test_requirements_file_exists(self):
        """Test that requirements.txt file exists."""
        requirements_path = Path("requirements.txt")
        assert requirements_path.exists(), "requirements.txt file is missing"

    def test_requirements_file_format(self):
        """Test that requirements.txt has valid format."""
        requirements_path = Path("requirements.txt")
        with open(requirements_path, "r") as f:
            lines = f.readlines()

        # Check that each line is either a comment or a valid package specification
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                # Should be in format package==version or package>=version etc.
                assert "==" in line or ">=" in line or "<=" in line or ">" in line or "<" in line, \
                    f"Invalid package specification: {line}"

    def test_python_version(self):
        """Test that Python version is compatible."""
        python_version = sys.version_info
        assert python_version.major == 3, "Python 3 is required"
        assert python_version.minor >= 7, "Python 3.7 or higher is required"

    def test_streamlit_installed(self):
        """Test that Streamlit is installed."""
        try:
            import streamlit
            assert True
        except ImportError:
            assert False, "Streamlit is not installed"

    def test_pandas_installed(self):
        """Test that Pandas is installed."""
        try:
            import pandas
            assert True
        except ImportError:
            assert False, "Pandas is not installed"

    def test_numpy_installed(self):
        """Test that NumPy is installed."""
        try:
            import numpy
            assert True
        except ImportError:
            assert False, "NumPy is not installed"

    def test_plotly_installed(self):
        """Test that Plotly is installed."""
        try:
            import plotly
            assert True
        except ImportError:
            assert False, "Plotly is not installed"

    def test_scipy_installed(self):
        """Test that SciPy is installed."""
        try:
            import scipy
            assert True
        except ImportError:
            assert False, "SciPy is not installed"

    def test_openpyxl_installed(self):
        """Test that openpyxl is installed (for Excel export)."""
        try:
            import openpyxl
            assert True
        except ImportError:
            assert False, "openpyxl is not installed"

    def test_salib_installed(self):
        """Test that SALib is installed (for sensitivity analysis)."""
        try:
            import SALib
            assert True
        except ImportError:
            assert False, "SALib is not installed"

    def test_install_requirements(self):
        """Test that requirements can be installed."""
        requirements_path = Path("requirements.txt")
        if requirements_path.exists():
            try:
                # Try to install requirements (this will fail if requirements are invalid)
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
                assert True
            except subprocess.CalledProcessError:
                assert False, "Failed to install requirements from requirements.txt"
        else:
            assert False, "requirements.txt file is missing"

    def test_all_required_packages_installed(self):
        """Test that all required packages are installed."""
        required_packages = [
            "streamlit",
            "pandas",
            "numpy",
            "plotly",
            "scipy",
            "openpyxl",
            "SALib"
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        assert len(missing_packages) == 0, f"Missing required packages: {', '.join(missing_packages)}"

    def test_package_versions(self):
        """Test that package versions are compatible."""
        # Test Streamlit version
        try:
            import streamlit
            assert streamlit.__version__ >= "1.0.0", "Streamlit version should be 1.0.0 or higher"
        except ImportError:
            assert False, "Streamlit is not installed"

        # Test Pandas version
        try:
            import pandas
            assert pandas.__version__ >= "1.0.0", "Pandas version should be 1.0.0 or higher"
        except ImportError:
            assert False, "Pandas is not installed"

        # Test NumPy version
        try:
            import numpy
            assert numpy.__version__ >= "1.0.0", "NumPy version should be 1.0.0 or higher"
        except ImportError:
            assert False, "NumPy is not installed"

        # Test Plotly version
        try:
            import plotly
            assert plotly.__version__ >= "4.0.0", "Plotly version should be 4.0.0 or higher"
        except ImportError:
            assert False, "Plotly is not installed"

        # Test SciPy version
        try:
            import scipy
            assert scipy.__version__ >= "1.0.0", "SciPy version should be 1.0.0 or higher"
        except ImportError:
            assert False, "SciPy is not installed"

    def test_package_imports(self):
        """Test that all required packages can be imported."""
        packages_to_test = [
            ("streamlit", "Streamlit"),
            ("pandas", "Pandas"),
            ("numpy", "NumPy"),
            ("plotly.graph_objects", "Plotly Graph Objects"),
            ("scipy.optimize", "SciPy Optimize"),
            ("openpyxl", "openpyxl"),
            ("SALib.sample", "SALib Sample"),
            ("SALib.analyze", "SALib Analyze")
        ]

        for package_path, package_name in packages_to_test:
            try:
                __import__(package_path)
            except ImportError:
                assert False, f"{package_name} cannot be imported"

    def test_package_functionality(self):
        """Test basic functionality of required packages."""
        # Test Streamlit basic functionality
        try:
            import streamlit as st
            assert hasattr(st, 'set_page_config')
            assert hasattr(st, 'file_uploader')
            assert hasattr(st, 'button')
            assert hasattr(st, 'tabs')
        except ImportError:
            assert False, "Streamlit basic functionality not available"

        # Test Pandas basic functionality
        try:
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            assert len(df) == 3
            assert 'a' in df.columns
            assert 'b' in df.columns
        except ImportError:
            assert False, "Pandas basic functionality not available"

        # Test NumPy basic functionality
        try:
            import numpy as np
            arr = np.array([1, 2, 3, 4, 5])
            assert len(arr) == 5
            assert arr[0] == 1
            assert arr[-1] == 5
        except ImportError:
            assert False, "NumPy basic functionality not available"

        # Test Plotly basic functionality
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            assert hasattr(fig, 'add_trace')
            assert hasattr(fig, 'update_layout')
        except ImportError:
            assert False, "Plotly basic functionality not available"

        # Test SciPy basic functionality
        try:
            from scipy.optimize import curve_fit
            def func(x, a, b):
                return a * x + b
            xdata = np.array([1, 2, 3])
            ydata = np.array([2, 4, 6])
            popt, pcov = curve_fit(func, xdata, ydata)
            assert len(popt) == 2
        except ImportError:
            assert False, "SciPy basic functionality not available"

        # Test openpyxl basic functionality
        try:
            import openpyxl
            wb = openpyxl.Workbook()
            assert wb.active.title == "Sheet"
            assert len(wb.sheetnames) == 1
        except ImportError:
            assert False, "openpyxl basic functionality not available"

        # Test SALib basic functionality
        try:
            from SALib.sample import saltelli
            from SALib.analyze import sobol
            problem = {
                'num_vars': 2,
                'names': ['x1', 'x2'],
                'bounds': [[0, 1], [0, 1]]
            }
            param_values = saltelli.sample(problem, 100)
            assert param_values.shape[0] > 0
            assert param_values.shape[1] == 2
        except ImportError:
            assert False, "SALib basic functionality not available"