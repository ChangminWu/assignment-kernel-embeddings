### Requirements
Code is written in Python (3.6).
External Library: spherecluster. Can be downloaded from https://github.com/clara-labs/spherecluster (minor modification were made to suit our codes)

### Datasets
Use the following link to download datasets: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
Extract the datasets into the "datasets" folder.

### Running the method
Use the following command: 

python main.py dataset method

where dataset is the name of the dataset, method is the employed kernel method (sp -> Embeddings-Optimal Assignment-Shortest Path, eoa -> Embeddings-Optimal Assignment).


### Examples
Example commands: 

python main.py MUTAG eoa
