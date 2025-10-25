# About This Repository

This repository is based on [v1_631](https://github.com/Biophysical-Fluorescence-Laboratory/Imaging_FCS.git) with additional fixes and modifications for specific use cases. It was created as a stable alternative while [v2](https://github.com/Biophysical-Fluorescence-Laboratory/Imaging_FCS.git) was under development. Note that experimental features from [v1_62_2](https://github.com/danielaik/Imaging_FCS_1_6x_GitHubRepo.git) (direct-camera-readout testing) are not up-to-date in this version. Useful features and improvements from this fork will eventually be integrated into v2, with v1 being phased out entirely.

# What is Imaging FCS

_Imaging FCS_ is an _ImageJ_ plugin featuring post-processing tools to calculate
and view spatio-temporal correlation functions from 16 bit grey tiff stack
files as well as data acquisition software for real-time image analysis. It was
written as a _FIJI plugin_

_Imaging FCS_ provides a comprehensive software tool to calculate and evaluate
spatiotemporal correlation functions. It includes the calculation of all auto-
or cross-correlation functions for arbitrary pixel binning and regions of
interest within an image, provides fit functions for total internal reflection
fluorescence (TIRF) and single plane illumination microscopy (SPIM) based FCS
measurements, can calculate the FCS diffusion laws and contains an essential
simulator to create simulated data for different diffusive modes.

ImagingFCS runs under ImageJ, FIJI and Micromanager, and it runs on PC, Linux,
and Mac OS. We will always use FIJI in the following text, but it should be
understood that the same is true for ImageJ and Micromanager.

# References

**(Deep learning)** Tang WH, et al. "Deep learning reduces data requirements
and allows real-time measurements in Imaging Fluorescence Correlation
Spectroscopy."
bioRxiv. 2023. https://doi.org/10.1101/2023.08.07.552352

**(Direct camera readout)** Aik DYK, et al. "Microscope alignment using
real-time Imaging FCS."
Biophys J. 2022. https://doi.org/10.1016/j.bpj.2022.06.009

**(GPU capabilities)** Sankaran J, et al. "Simultaneous spatiotemporal
super-resolution and multi-parametric fluorescence microscopy."
Nat Commun. 2021. https://doi.org/10.1038/s41467-021-22002-9

**(Correlator scheme)** Sankaran K, et al. "ImFCS: a software for imaging FCS
data analysis and visualization."
Opt Express. 2010. https://doi.org/10.1364/OE.18.025468

## Disclaimer

The software and data on this site are provided for personal or academic use
only and may not be used in any commercial venture or distributions. All files
have been virus scanned, however, for your own protection; you should scan
these files again. You assume the entire risk related to your use of this
software and data. By using the software and data on this site your expressly
assume all risks of data loss or damage alleged to have been caused by the
software and data. The Biophysical Fluorescence Laboratory at NUS is providing
this data "as is," and disclaims any and all warranties, whether express or
implied, including (without limitation) any implied warranties of
merchantability or fitness for a particular purpose. In no event will the
Biophysical Fluorescence Laboratory at NUS and/or NUS be liable to you or to
any third party for any direct, indirect, incidental, consequential, special or
exemplary damages or lost profit resulting from any use or misuse of this
software and data.
