# scGAAE: denoising scRNA-seq data using a deep learning-based graph attention auto-encoder

Here, we propose a novel imputation method, scGAAE, based on a deep learning driven graph attention auto-encoder. scGAAE stacks multiple encoder/decoder layers equipped with graph attention mechanisms to reconstruct both the graph structure and the node features directly.Additionally, scGAAE does not require assumptions about the distribution or low rank properties of the data. It imputes the dropout values while preserving the original expression features as much as possible.

**Requirment**<br>
Python version: 3.7<br>
conda create -n Python_3.7_scGAAE python==3.7<br>
conda activate Python_3.7_scGAAE<br>

pip install matplotlib<br>
pip install scipy<br>
pip install tensorflow==1.13.1<br>
pip install protobuf==3.20.3<br>
pip install networkx<br>
pip install seaborn<br>
pip install anndata<br>
pip install scanpy<br>

**Please note**:<br>
1、Find and open *"/home/../miniconda3/envs/Python_3.7_scGAAE/lib/python3.7/site-packages/umap/`__init__.py`"*, <br>
change `"from importlib.metadata import version, PackageNotFoundError"` <br>
    to `"from importlib_metadata import version, PackageNotFoundError"` on line 37 of `__init__.py`, and then save it. <br>

2、Or you can copy and paste the `__init__.py` from the this repository into the umap folder to replace the `__init__.py`<br>

**Program usage**<br>
You can set parameters in the scGAAE.py and run it using：<br> 
`pyhon scGAAE.py`<br>
