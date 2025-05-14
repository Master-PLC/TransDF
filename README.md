# TransDF: Time-series Forecasting Needs Transformed Label Alignment


<h3 align="center">Welcome to TransDF</h3>

<p align="center"><i>Enhancing Time-series forecasting performance with simple transformation.</i></p>


The repo is the official implementation for the paper: [TransDF: Time-series Forecasting Needs Transformed Label Alignment](https://openreview.net/forum?id=RxWILaXuhb).

We provide the running scripts to reproduce experiments in `/scripts`, which covers two mainstream tasks: **long-term forecasting and short-term forecasting**.

🚩**News** (2025.5) The implementation of TransDF is released, with scripts on two tasks.

## Usage

0. Add a PCA decomposition step while loading training data.
```python
from utils.polynomial import get_pca_base

class Dataset_ETT_hour_PCA(Dataset_ETT_hour):
    def __init__(
        self, rank_ratio=1.0, pca_dim="T", reinit=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.pca_fit(rank_ratio, pca_dim, reinit)

    def pca_fit(self, rank_ratio=1.0, pca_dim="T", reinit=1):
        if self.set_type != 0:
            # Note: we only apply PCA transformation on train data
            self.pca_components = None
            return

        print("Fitting PCA ...")
        label_seq = []
        for i in range(self.__len__()):
            _, label, _, _ = self.__getitem__(i)
            label = label[-self.pred_len:]
            label_seq.append(label)
        label_seq = np.array(label_seq)  # shape: [N, P, D]
        # Note: get pca projection basis for pytorch based projection
        self.pca_components, self.initializer, self.weights = get_pca_base(
            label_seq, rank_ratio, pca_dim, reinit
        )
        print(f"PCA components shape: {self.pca_components.shape}")
        print(f"PCA weights shape: {self.weights.shape}")
```

1. Implement TransDF by adapting the following script in your pipeline
```python
from utils.polynominal import Basis_Cache

pca_cache = Basis_Cache(
    train_data.pca_components, train_data.initializer, weights=train_data.weights, device='cuda'
)

# The canonical temporal loss
loss_tmp = ((outputs - batch_y)**2).mean()

# The proposed transformed loss
kwargs = {
    'pca_dim': 'T', 'pca_cache': pca_cache, 'use_weights': 0, 
    'reinit': 1, 'device': 'cuda'
}
loss_trans = (pca_torch(outputs, **kwargs) - pca_torch(batch_y, **kwargs)).abs().mean()
# Note. The transformed loss can be used individually or fused with the temporal loss using finetuned relative weights.
```

2. Install Python 3.10 and pytorch 2.4.0. For convenience, execute the following commands.

```bash
conda create -n transdf python=3.10
conda activate transdf

# we recommend using conda to install pytorch and torch-geometric dependencies
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia

# if failed, try to install pytorch and torch-geometric dependencies using pip
# pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

pip install https://data.pyg.org/whl/torch-2.4.0%2Bcu118/torch_cluster-1.6.3%2Bpt24cu118-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.4.0%2Bcu118/torch_scatter-2.1.2%2Bpt24cu118-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.4.0%2Bcu118/torch_sparse-0.6.18%2Bpt24cu118-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.4.0%2Bcu118/torch_spline_conv-1.2.2%2Bpt24cu118-cp310-cp310-linux_x86_64.whl

pip install -r requirements.txt
```

3. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder `./dataset`. Here is a summary of supported datasets.

<p align="center">
<img src=".\pic\dataset-fredf.jpg" height = "200" alt="" align=center />
</p>

4. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```bash
# long-term forecast
bash ./scripts/long_term_forecast/Fredformer.sh
bash ./scripts/long_term_forecast/iTransformer.sh
bash ./scripts/long_term_forecast/FreTS.sh
bash ./scripts/long_term_forecast/MICN.sh

# short-term forecast
bash ./scripts/short_term_forecast/Fredformer.sh
```

5. Apply TransDF to your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Fredformer.py`.
- Include the newly added model in the `./models/__init__.py.MODEL_DICT`.
- Create the corresponding scripts under the folder `./scripts`. You can follow `./scripts/long_term_forecast/Fredformer.sh`.


## Acknowledgement

This library is mainly constructed based on the following repos, following the training-evaluation pipelines, the implementation of baseline models, transformation implementation and label alignment methods:

- Time-Series-Library: https://github.com/thuml/Time-Series-Library.
- Dilate: https://github.com/vincent-leguen/DILATE.
- Soft-DTW: https://github.com/Maghoumi/pytorch-softdtw-cuda.
- RobustPCA: https://github.com/loiccoyle/RPCA.

All the experiment datasets are public, and we obtain them from the following links:
- Long-term Forecasting and Imputation: https://github.com/thuml/Autoformer.
- Short-term Forecasting: https://github.com/ServiceNow/N-BEATS.



