## Installation

1. Create a virtualenv (python 3.ww)
   `pip3 install virtualenv`
   `python3 -m venv env`
   `source env/bin/activate`
   maybe need upgrade `python3 -m pip install --upgrade pip`
2. Install dependencies inside of virtualenv (`pip install -r requirements.pip`)
3. Download the dataset in following link [Dataset](https://lqdtueduvn-my.sharepoint.com/:f:/g/personal/phong_tt_lqdtu_edu_vn/EnsENFazD2FNrs3RZCKRgqcBfalgtfJBIttXd1mkSu7lZg) and copy `data` folder with correct name: cic-ids, nb-iot, NSLKDD, unsw, spambase,...
   If read dataset failed, please check the expected path in `function/datasets/data_load/<data_name>.py`

### Run

Before you can run any experiments, you have to:

1. Check and update (if needed) environment setup in file `function/arguments.py`.
2. Check and update (if needed) config for each dataset in file `config/<dataset>.yaml`.
3. Check and update (if needed) expected dataset in file `run.sh`.
4. Run experiments: `bash run.sh`.
