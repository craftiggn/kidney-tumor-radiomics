# kidney-tumor-radiomics
Extraction of radiomic features from segmentated kidney tumors.

## Grading system
The grading of the tumor will be done using RPN (Radius, Position of tumour, iNvasion of renal sinus), as described in: https://pmc.ncbi.nlm.nih.gov/articles/PMC10397239/ or on the figure below.
![alt text](figures/RPN_chatrt.jpg)

### Radius
Can be done using a regular algorithm, no need for AI.
Need to confirm it works when several tumors are present.

### Position of tumor
Would be some what hard to describe - use ML
- should the position be described with more nuance to the algorithm?

### Invasion of renal sinus
Looks like it could be hard coded in, but in my opinion an ML appraoch seems more appropriate.

## Data annotation
Read the data with "nibabel"

Label the data using 3d slicer, 

# ToDo
- message kits23 winners for annotation model access

# Notes
sparse matrix architecture - rejected due to poor implemenation/availability in tf.keras
