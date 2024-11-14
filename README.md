# Juo Lab Line Scanner

This Python application, developed by Sahil Suresh in 2022, is designed for Juo Lab at Tufts University. It facilitates line scans and peak analysis of fluorescence intensity in images, with user-defined settings and group configurations. The application uses PySimpleGUI for its GUI, OpenCV for image processing, and SciPy for peak detection and analysis.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Functionality](#functionality)
  - [Settings](#settings)
  - [Groups](#groups)
  - [Main Functions](#main-functions)
- [Export Options](#export-options)
- [Dependencies](#dependencies)
- [Credits](#credits)
- [License](#license)

---

## Features

- **Image Loading**: Load single images or batch-process folders of images.
- **Line Scan and ROI Scan**: Perform intensity line scans and Region of Interest (ROI) scans on selected areas.
- **Peak Analysis**: Analyze fluorescence intensity profiles using SciPy or Eli Billauer’s peak detection formula.
- **Data Visualization**: Display intensity data, cumulative distribution functions, and bar charts for group comparisons.
- **Customizable Settings**: Adjust line width, image scale, analysis parameters, and autoscaling options.
- **Group Analysis**: Organize data into groups and calculate average intensities across groups.
- **Data Export**: Export analyzed data, figures, and results in `.png` and `.json` formats.

---

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/YOUR-USERNAME/JuoLabLineScanner.git
    cd JuoLabLineScanner
    ```

2. **Install Dependencies**
    - It is recommended to use a virtual environment:
      ```bash
      python -m venv env
      source env/bin/activate  # On Windows use `env\Scripts\activate`
      ```
    - Install required packages:
      ```bash
      pip install -r requirements.txt
      ```

3. **Run the Application**
    ```bash
    python juo_lab_line_scanner.py
    ```

---

## Usage

### Main Menu Options

- **File**:
  - **Select Folder**: Choose a folder of images for batch processing.
  - **Select Image**: Load a single image for analysis.
  - **Export**: Export data and figures in `.png` and `.json` formats.
- **Settings**:
  - **User Preferences**: Access and customize various settings for line width, scaling, and analysis.
- **Tools**:
  - **Edit Groups**: Configure groups for organizing and analyzing images by labels and color codes.
  - **ROI Scan**: Perform a Region of Interest scan on a selected area of the image.
- **Help**:
  - **About**: Display application information.
  - **User Guide**: Reference for using the app.

### Working with Images

1. **Load an Image or Folder**: Use **Select Image** or **Select Folder**.
2. **Line Scan**: Select **Single Line Scan** to perform a scan on the loaded image, or **Batch Line Scan** for batch processing.
3. **ROI Scan**: Use the **ROI Scan** tool to analyze intensity within a user-defined area of the image.
4. **Save Results**: Use the **Export** option to save results, data, and figures.

---

## Functionality

### Settings

Settings can be customized to control the appearance and behavior of the application:
- **LINEWIDTH**: Thickness of the line used in scans.
- **IMAGESCALE**: Scale of the loaded image.
- **PEAKDETDELTA**: Delta for Eli Billauer’s peak detection.
- **YMIN/YMAX**: Limits for Y-axis in plots.
- **AUTOSCALE**: Auto-adjust Y-axis.
- **SCIPYANALYSIS**: Use SciPy’s peak detection method.
- **STANDARDIZE**: Standardize intensity data.
- **SCIPYMINDISTANCE/SCIPYHEIGHT/SCIPYTHRESHOLD**: Parameters for SciPy peak detection.

### Groups

Group settings allow for organizing scans into labeled groups:
- **GROUPNUMBER**: Number of groups.
- **COLORLABELS**: Set colors for each group.
- **GROUPLABELS**: Assign labels to each group.

### Main Functions

#### Line Scan Tools

- **Single Line Scan**: Perform a line scan on the loaded image and analyze intensity profile.
- **Batch Line Scan**: Perform line scans on multiple images within a selected folder.

#### Analysis Tools

- **Peak Analysis**: Calculate peak intensity, baseline intensity, peak distance, and width.
- **SciPy Analysis**: Detect peaks using SciPy’s `find_peaks` function.
- **Eli Billauer Analysis**: Detect peaks using Eli Billauer’s peak detection method.

#### Group Analysis

- Perform statistical analysis across different groups. The app calculates average peak intensities and error bars for each group.

---

## Export Options

Export data and figures to various formats:
- **Figures**: Export as `.png` files.
- **Data**: Export batch scan results as `.json`.

---

## Dependencies

This application uses the following libraries:
- `PySimpleGUI`
- `OpenCV` (`cv2`)
- `NumPy`
- `Matplotlib`
- `SciPy`
- `Pandas`
- `Keyboard`
- `Skimage`

---

## Credits

- **Author**: Sahil Suresh
- **Contributors**: Eli Billauer (Peak Detection function/formula)

---

## License

This project is licensed under the MIT License.
