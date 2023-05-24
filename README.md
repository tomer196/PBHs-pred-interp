# PBHs prediction and Interpretability
Interpretable deep learning was used to identify structure-property relationships governing the HOMO-LUMO gap and relative stability of polybenzenoid hydrocarbons (PBHs). To this end, a ring-based graph representation was used. In addition to affording reduced training times and excellent predictive ability, this representation could be combined with a subunit-based perception of PBHs, allowing chemical insights to be presented in terms of intuitive and simple structural motifs. The resulting insights agree with conventional organic chemistry knowledge and electronic-structure based analyses, and also reveal new behaviors and identify influential structural motifs. In particular, we evaluated and compared the effects of linear, angular, and branching motifs on these two molecular properties, as well as explored the role of dispersion in mitigating torsional strain inherent in non-planar PBHs. Hence, the observed regularities and the proposed analysis  contribute to a deeper understanding of the behavior of PBHs and form the foundation for design strategies for new functional PBHs. 

<p align="center">
<img src="https://github.com/tomer196/PBHs-pred-interp/blob/main/Interp-example.png" width="400" >
</p>

## Setup
1. Clone this repository by invoking
```
git clone https://github.com/tomer196/PBHs-pred-interp.git
```
2. Download dataset (`csv` + `xyz`s) from [COMPAS](https://gitlab.com/porannegroup/compas)
3. Update `csv` + `xyz`s paths in `utiles/args.py`
4. Install conda environment. The environment can be installed using the `environment.yml` by invoking
```
conda env create -n PBHs --file environment.yml
```
Alternatively, dependencies can be installed manually as follows:
```
conda create -n PBHs python=3.8
conda activate PBHs
conda install pytorch=1.10 cudatoolkit=10.2 -c pytorch
conda install -c dglteam dgl-cuda10.2
conda install numpy matplotlib networkx scipy tensorboard pandas 
pip install requests
```

## Usage
### Training
The training script will train the model using the train and validation datasets. 
When training finish will run evaluation on the test set and will print the results. 
The saved model and the some plots will be save in `summary/{exp_name}`. 
```
python train.py --name 'exp_name' --target_features 'GAP_eV, Erel_eV' 
```

target_features should be separate with `,`.  
Full list of possible arguments can be seen in `utiles/args.py`.  

### Evaluation
Run only the evaluation on trained model.
```
python eval.py --name 'exp_name'
```

### Interpretability
Running the interpretability algorithm. Will save all the molecules in the dataset 
with their GradRAM weights in the same directory as the logs and models.
```
python interpretability_save_all.py --name 'exp_name'
```
To run only on subset of molecules run `interpretability_from_names.py` and change the molecules name
list in line 27.

## Repo structure
- `data` -  Dataset and data related code.
- `se3_trnasformers` - SE3 model use as a predictor
- `utils` - helper function. 
- `summary` - Logs, trained models and interpretability figures for each experiment. 


