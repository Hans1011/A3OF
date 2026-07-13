These code files contain the implementation of the closed-loop autonomous experimentation framework in A3OF, including the Bayesian optimization algorithm as the active learning strategy and the control and communication codes for the global operation modules (GOMs) and local functional modules (LFMs).

Main.py: The central control program that integrates Bayesian optimization, GOMs and LFMs operation as well as hardware communication to establish the fully automated closed-loop experimentation pipeline.

Optimizer.py: The Bayesian optimization module that functions as the active learning engine of A3OF. It analyzes historical experimental data and reward values, and autonomously recommends subsequent water-oil recipes within the six-dimensional parameter space for iterative optimization.

OT-2_control.py: The GOM liquid handling control module that operates the OT-2 pipetting system to automatically prepare experimental reagents, perform dilution procedures, and dispense aqueous phase, oil phase, MNP solution, and other reagents into the microfluidic chip.

Pre-detection.py: The LFM pre-detection module that performs automated oil-surfactant compatibility testing through image-based analysis. It identifies incompatible interfaces using computer vision algorithms and eliminates invalid formulations before downstream MNP translocation experiments.

Characterization.py: The LFM characterization module that performs automated analysis of MNP translocation outcomes. It extracts morphological features from captured images, calculates quantitative reward values based on translocation quality, and provides feedback to the Bayesian optimization algorithm for subsequent experimental recommendations.