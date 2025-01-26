# Sampling Assignment

## GitHub Repository

This repository contains the code and solution for the Sampling Assignment.

## Requirements

The program needs to perform the following tasks:

1. Download the data-set from the given link:
   `https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv`

2. Convert the data-set into a balanced class data-set using the techniques discussed in the class.

3. Create five samples using the sample size detection formula as discussed in the class.

4. Apply five different sampling techniques (Sampling1, Sampling2, Sampling3, Sampling4, Sampling5) on five different ML models (M1, M2, M3, M4 and M5).

5. Determine which sampling technique gives higher accuracy on which model.

6. Put the solution on the "GitHub" with discussion and then submit the "GitHub" link using the submission link.

## Solution

The solution is implemented in the `sampling_102203810.py` file. Here's a summary of the steps:

1. Download the data from the provided link.
2. Balance the class data using SMOTE oversampling.
3. Create 5 samples using the sample size formula.
4. Apply 5 sampling techniques on 5 ML models.
5. Evaluate the performance of each sampling technique on each model.
6. Determine the best sampling technique for each model.
7. Upload the solution to GitHub and submit the link.

The specific implementation details are provided in the Python script.

**Model | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5**
--- | --- | --- | --- | --- | ---
M1 | 0.83606557377049 | 0.89130434782609 | 0.88311688311688 | 0.91803278688525 | 0.91304347826087
M2 | 0.88524590163934 | 0.90217391304348 | 0.87012987012987 | 1.0 | 0.95652173913043
M3 | 0.96721311475410 | 1.0 | 1.0 | 1.0 | 0.96739130434783
M4 | 0.55737704918033 | 0.68478260869565 | 0.57142857142857 | 0.72131147540984 | 0.65217391304348
M5 | 0.73770491803279 | 0.77173913043478 | 0.71428571428571 | 0.77049180327869 | 0.80434782608696

## Usage

1. Clone the repository: `git clone https://github.com/AnMaster15/sampling.git`
2. Run the script: `python sampling_102203810.py`
3. The results will be stored in the GitHub repository, and the link can be submitted.

## Conclusion

The assignment covers the essential aspects of sampling techniques and their application to various ML models. The solution provides a comprehensive approach to addressing the problem and determining the most effective sampling technique for each model.
