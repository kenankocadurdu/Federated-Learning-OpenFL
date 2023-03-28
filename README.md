# An Open-Source Framework For Federated Learning (OpenFL)
## Federated Learning
Federated Learning is a machine learning technique that focuses on data privacy, proposed by MacMahan in 2016. McMahan defined unified learning as a collaborative model built with distributed data on mobile devices. Today, this technique is also frequently used in areas where data is sensitive, such as medicine, finance, and shopping.

## OpenFL
OpenFL is a Python 3 framework for Federated Learning. OpenFL is designed to be a flexible, extensible and easily learnable tool for data scientists. This framework currently offers two ways to set up and run experiments with a federation: the Director-based workflow and Aggregator-based workflow.

### Director Based Workflow
#### 1. Create a virtual environment, upgrade pip, and install OpenFL
```
apt-get install python3-venv -y
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install openfl
#torch package installation convenient for your system
#pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 2. Open 4 Different Terminal
##### 2.1. First Terminal (run the Director)
```
source venv/bin/activate
cd ./server_director/director/
bash start_director.sh
```
##### 2.2. Second Terminal (connect to Director)
```
source venv/bin/activate
cd ./server_director/envoy/
bash start_envoy.sh
```
##### 2.3. Third Terminal (connect the Director)
```
source venv/bin/activate
cd ./server_client/envoy/
bash start_envoy.sh
```
##### 2.4. Last Terminal (run the training)
```
source venv/bin/activate
cd ./server_director/workspace/
jupyter lab train.ipynb
```
Restart Kernel and Run All Cells
