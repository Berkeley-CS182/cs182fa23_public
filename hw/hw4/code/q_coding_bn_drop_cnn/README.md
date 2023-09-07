## Setup
Make sure your machine is set up with the assignment dependencies.

**[Option 1] Use Google Colab (Recommended):**
The preferred approach to do this assignment is to use [Google Colab](https://colab.research.google.com/).
You need to use a GPU-based environment for the `hw3_pytorch_cnn.ipynb` notebook, see instructions in that notebook
to switch to a GPU runtime.


**[Option 2] Use a local Conda environment:**
This approach is **not recommended** because the GPU of most PCs or laptops does not meet the requirements of
training deep learning models.

**Download data:**
Once you have the starter code, you will need to download the CIFAR-10 dataset.

```bash
cd deeplearning/datasets
./get_datasets.sh
cd ../..
```

If you are on Mac, this script may not work if you do not have the wget command
installed, but you can use curl instead with the alternative script.
```bash
cd deeplearning/datasets
./get_datasets_curl.sh
cd ../..
```

**Start Jupyter:**
After you have the CIFAR-10 data, you should start the IPython notebook server
from this directory.
```bash
jupyter notebook
```
