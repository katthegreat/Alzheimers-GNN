# Alzheimer's GNN Project

Gauri Saraf and Gunn Madan

## What we are doing

We are using graph neural networks to classify Alzheimer's disease and normal controls using brain functional connectivity data

Each brain region is a node and the correlations between regions are the edges

We want to compare three ways of representing the brain connections

1. Regular correlation graph
2. Positive and negative correlation graphs
3. Four separate graphs based on correlation strength

For the four graph approach we are using

- High positive r > 0.5
- Low positive 0 < r <= 0.5
- High negative r < -0.5
- Low negative -0.5 <= r < 0

We plan to use GCNs and compare the different graph representations for AD vs NC classification

We also plan to use contrastive learning before supervised training

## Dataset

We originally looked at ADNI and OASIS-3

The problem with ADNI was that the main download page mostly gave us raw fMRI scans and other imaging data instead of the precomputed correlation matrices we needed

We did not want to spend time preprocessing MRI scans since our project focuses on graph construction and classification

So I found a GitHub repository called MNMI_AD that contains processed ADNI functional connectivity matrices

https://github.com/Guoshi-Li/MNMI_AD

This means we are still using data derived from ADNI but we can work directly with the connectivity matrices

We still need a second dataset and OASIS-3 is one possibility if we can find precomputed matrices

## Setting up the project

Gunn when you open the project in VS Code make sure you are inside the Alzheimers-GNN folder

Create the virtual environment

python3.11 -m venv .venv

Activate it on Mac

source .venv/bin/activate

For Windows PowerShell

.venv\Scripts\Activate.ps1

The terminal should show (.venv) when it is activated

Install the packages

python -m pip install numpy pandas scipy openpyxl

numpy is for working with matrices

pandas is for reading participant information

scipy is for loading MATLAB files

openpyxl is for reading Excel spreadsheets

We will install PyTorch and PyTorch Geometric when we start working on the GNN model

## Downloading the dataset

Run these commands from the main project folder

mkdir -p data/raw

git clone https://github.com/Guoshi-Li/MNMI_AD.git data/raw/MNMI_AD

This downloads the dataset into data/raw/MNMI_AD

To check the downloaded files

ls data/raw/MNMI_AD

The main folders are

DATA
SUBJECT
MNMI
ANALYSIS
OUTPUT

Inside DATA there are three folders

FC contains functional connectivity matrices

SC contains structural connectivity matrices

BOLD contains brain activity time series

We are mainly using FC

To see the FC files

find data/raw/MNMI_AD/DATA/FC -type f | head -20

Some of the files are

FC_AD19.mat
FC_AD31.mat
FC_NC7.mat
FC_NC12.mat
FC_MCI11.mat

AD means Alzheimer's disease

NC means normal control

MCI means mild cognitive impairment

For now we are focusing on AD and NC

## Participant information

The participant spreadsheet is located at

data/raw/MNMI_AD/SUBJECT/MNMI_SUBJECT_INFO.xlsx

I checked the spreadsheet using pandas

It has 144 rows and 15 columns

Some useful columns are

PTID for participant ID

DX for diagnosis

AGE for age

SEX for sex

MMSE for cognitive assessment score

SCAN_DATE for scan date

We plan to use NC as 0 and AD as 1

We still need to verify how the matrix filenames match the participant IDs

## Checking the correlation matrices

I loaded one of the FC files using scipy

The file was FC_AD19.mat

The output was

Variable name FC

Shape 46 x 46

Data type float64

Minimum value -0.6161

Maximum value 1.0

Contains negative values True

Symmetric True

Diagonal values 1

This means the file contains a 46 x 46 connectivity matrix with both positive and negative values

The 46 rows and columns represent 46 brain regions

The matrix is symmetric because the correlation between region A and B is the same as B and A

This is the type of data we need for our graph construction code

We have only checked one matrix so far and still need to validate the others

## Code so far

The main files are

src/inspect_data.py

src/graph_construction.py

inspect_data.py was used during the initial dataset investigation

graph_construction.py currently tests how we separate a correlation matrix into four graphs

I tested the graph construction code using a small example with five brain regions

To run it

python src/graph_construction.py

The output showed

High positive graph 6 directed edges

Low positive graph 4 directed edges

High negative graph 4 directed edges

Low negative graph 6 directed edges

The edge indices show which brain regions are connected

The edge weights contain the correlation values

For example 0.8 goes into the high positive graph and -0.7 goes into the high negative graph

The code worked with the example matrix

We have not connected the real MNMI_AD matrices to the graph construction script yet

## Loading an actual FC matrix

Gunn you can test one of the real matrices using this code in Python

from scipy.io import loadmat
import numpy as np

data = loadmat("data/raw/MNMI_AD/DATA/FC/FC_AD19.mat")

matrix = data["FC"]

print(matrix.shape)
print(np.min(matrix))
print(np.max(matrix))
print(np.any(matrix < 0))
print(np.allclose(matrix, matrix.T))

This checks the matrix dimensions and whether it contains positive and negative correlations

## What we have done so far

- Set up the project folders and virtual environment
- Installed the basic Python packages
- Looked through OASIS-3 and ADNI
- Found the MNMI_AD processed dataset
- Downloaded the MNMI_AD repository
- Found the functional connectivity files
- Loaded the participant spreadsheet
- Confirmed that one FC matrix is 46 x 46
- Confirmed that the matrix contains negative correlations
- Tested the four graph construction approach using an example matrix

## What we need to do next

1. Check all FC files to make sure they have valid matrices
2. Confirm the number of AD and NC participants
3. Verify how the matrix files match participant information
4. Update graph_construction.py to use the real FC matrices
5. Create graph data objects for PyTorch Geometric
6. Implement the regular graph and positive negative graph baselines
7. Implement the four graph GCN model
8. Add contrastive pretraining
9. Train and evaluate the models
10. Find a second dataset with precomputed connectivity matrices
11. Compare the results across the graph representations and datasets

The main thing completed so far is finding usable functional connectivity data and testing the graph construction logic

We have not trained the actual GNN yet
