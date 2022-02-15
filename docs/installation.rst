============
Installation
============

0. Virtual environments are strongly recommended, to prevent dependencies with conflicting versions. Create a conda virtual environment and activate it:

::

$conda create -n n2j python=3.8 -y
$conda activate n2j

1. Clone the repo and install.

::

$git clone https://github.com/jiwoncpark/node-to-joy.git
$cd node-to-joy
$pip install -e . -r requirements.txt

2. (Optional) To run the notebooks, add the Jupyter kernel.

::

$python -m ipykernel install --user --name n2j --display-name "Python (n2j)"
