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
- 458 - to nie guz nerki a miedniczki czyli zupełnie inny nowotwór
- 466 - to samo  - inny nowotwór

# Notes
WARNING: 5 cases are annotated but missing from valid_cases: ['case_00001', 'case_00002', 'case_00006', 'case_00008', 'case_00015']
- 01 - 2 masses
- 02 - 3 masses
- 06 - 3 masses
- 08 - many masses
- 15 - 3 masses

Maybe we can use tumors with multiple cases, why not?
- Need to figure out how masks work in more detail

Tumours may be noninvasive, semi-invasive, or invasive. Noninvasive tumours are partly or predominantly exophytic peripheral tumours and have a clear rim of normal parenchyma between the tumour and the renal sinus. Semi-invasive tumours include three distinct types: (1) peripheral tumours touching the renal sinus; (2) completely endophytic tumours confined within the renal parenchyma; and (3) hilar tumours that are superficial and not invading the renal sinus. Invasive tumours are those that extend into the renal sinus. Thus, peripheral tumours can be noninvasive, semi-invasive, or invasive. Hilar tumours can be semi-invasive or invasive, and all central tumours are exclusively invasive (Fig. 1).

# Implementation

## Launching
after geting the repo, kits23 and the raw annotation from google drive (.xlsx) ready run:
1. `gen_single_tumor_index.py` - to genereate `data/single_tumor_cases.json`
2. `convert_kits_dataset.py` - to generate `/home/marcel/Repos/Data/kits23_converted/` directory and fill it with data prepared for training i.e. segmentations as .npy arrays and labels.csv.

### Updated
after geting the repo, kits23 and the raw annotation from google drive (.xlsx) ready run:
1. 

## Input standardisation ideas
create a bounding box for the kidney-tumor system, 

### Helpfull tensorflow functions
use one-hot encoding for input? - https://www.tensorflow.org/api_docs/python/tf/one_hot
resize the input image? - https://www.tensorflow.org/api_docs/python/tf/image/resize
use TFRecord for data representation - https://www.tensorflow.org/tutorials/load_data/tfrecord
use tf.data to read TFRecord 


### later adition to this function 
- create synthetic data
- split test - train set taking into account the probablilty of all properties
- 

# Questions
- jak powinny być przygotowane dane do uczenia maszynowego na icm, w jakiej strukturze je zapisać?
- TFRecord, generator vs. preloaded zrobić benchmark generatora, może nie ma sensu jeśli się wczytuje w ciągu sekund to nie ma co się pocić
dataset.save

- jakieś rady co do uruchamiania uczenia na rysach aby była dobra wydajność (nvm. batch size większy może?), jak to ocenić czy dorze działa?
32 lub mniej

- one hot na inpucie? to dane wtedy będą 4 wymiarowe + batch
nie

- jakie biblioteki wykorzystywać do tworzenia syntetycznych danych: keras.layers jak na ćwiczeniach, czy własne funkcje tf?
keras.layers będzie lepiej/ławiej - może własną warstwę zaprojektować 

- jak zrobić dobry pipeline?
czasami można na żywo jak będzie szybkie

- da się wizualizować dane zapisane w ten sposób w miarę bez-problematyczny sposób?
tak

- czy są jakieś sposoby na automatyczny tuning hyperparametrów, coś w kerasie? czy muszę to ręcznie robić?
sklearn - pakiet do optymalizacj

- jakieś inne rady przy tworzeniu modeli które mają później być użyte w aplikacji (format zapisu lub coś)
tf.keras.save

