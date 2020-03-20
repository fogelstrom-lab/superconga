# Fermi surface plotting tools

## External packages required
Python3 should already be installed on your system, but we also need virtualenv. On Ubuntu install with
```
sudo apt install virtualenv
```
We use this to enforce the same package versions for the entire team.

## Setup the virtual environment
To create a virtual environment write in the folder you want to create it, i.e. tools:
```
virtualenv -p python3 venv
```
This will create a python3 environment in the folder venv. To activate it write
```
source venv/bin/activate
```
When activated install all our dependencies with
```
pip install -r requirements.txt
```
This only needs to be done once.

To deactivate the virtual environment write:
```
deactivate
```

## Usage
Remember to activate it with the command above. You can then run the scripts, e.g.
```
./run_fs.sh
python plot_fermi_surface.py fs1 fs2
```
