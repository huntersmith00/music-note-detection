# Music Note Detection

## Overview

This project explores several approaches to musical pitch detection in both **real-time audio streams** and **pre-recorded audio**. The goal is to evaluate different DSP and ML (Planned) techniques for identifying fundamental frequencies and, ultimately, for classifying higher-level musical context (key, mode, harmonic qualities, etc.).

This repository currently contains early experiments, test scripts, and prototype implementations. Some methods (e.g., FFT-based detection) are still unstable or under refinement.

## Current Experimental Methods

### **1. Fast Fourier Transform (FFT) Pipeline — *Prototype / Unstable***

* Implements a windowed FFT for frequency-domain analysis
* Currently inaccurate due to noise sensitivity and fundamental-harmonic confusion
* Includes test scripts (`testFFT.py`) documenting issues and debugging attempts

### **2. Harmonic Product Spectrum (HPS) — *Partially Functional***

* Downsampling + product-of-spectra approach
* More stable than FFT for clean single-note input
* Provides usable pitch estimates under controlled conditions
* Code partially adapted/AI-assisted, then modified and tested

### **3. Autocorrelation — *Planned***

* Intended for robust time-domain pitch detection
* Not yet implemented; design notes included

### **4. Machine Learning (CREPE) — *Concept Stage***

Exploring use of ML models for recognizing:

* Key / tonality
* Mode
* Time signature
* Basic harmonic context (e.g., major/minor quality)
* Possibly chord classification

These components are purely exploratory at the moment.

## Project Goals

Short term:

* Compare FFT, HPS, and autocorrelation accuracy
* Improve stability of real-time pitch detection
* Visualize frequency-domain and time-domain signals for debugging

Long term:

* Real-time pitch detection with low latency
* Classify higher-level music theory properties
* Explore lightweight ML models for classification tasks
* Integrate the pipeline into a usable tool or UI

## Status

This repository is an **active prototype and research sandbox**, not a finished product.
Methods vary in completeness, and some test files reflect ongoing debugging.
