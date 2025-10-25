# Changelog

All notable changes to this repo will be documented in this file

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [1.6.4da] - 2025-10-06

### Feature

- Allow batch analysis to start with user defined first and last frame to correlate
- Batch PSF analysis: disabled dialog box to allow as many PSF permutations

### Fixed

- Fix batch process loading and writing path to tiff file
- Fix load new image button
- Fix ccf q map in binning mode now loaded correctly
- Load xlsx file reading value row-by-row minus dccf and nnb
- Saved pixel information row-by-row to bypass 128x128 column limit in xlsx

### Tested

CLI compilation notes:

    mvn clean

    mvn clean package -DgenerateJniHeaders=true

    mvn clean package -DcompileLibs=true -e

    mvn test

## [pre 1.6.4da] - 2024-04-05

- Essentially a copy from https://github.com/Biophysical-Fluorescence-Laboratory/Imaging_FCS.git **c5a25a0387e3c3be324260dfb42ca6bc10c38c4c**. Changes from https://github.com/danielaik/Imaging_FCS_1_6x_GitHubRepo.git are not updated. Some feature are not tested especially dcr.
- Starting from CMake 3.18, the FindCUDA module has been deprecated. The modern CMake approach involves using the find_package(CUDAToolkit) module, which replaces FindCUDA. Update all the CMakeLists.txt files
- For modern Cmake use target_include_directories instead of include_directories for visibility within agpufit project for example
