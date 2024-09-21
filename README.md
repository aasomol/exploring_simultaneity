This file contains all the necessary explanations to run our experiments. In "experiments" file, there are located all the problems and plans of our analysis. 
# Setup

We first need to install pddlgym. For this, just go inside pddlgym folder (the first one) and run:

pip install -e .

This needs to be done this way because we slightly modified PDDLGym to fit our work.

Necessary extra installs:

- Pytorch, torchvision and torchaudio (refer to https://pytorch.org/)
- Pytorch-geometric (refer to https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

We include all the packages that are installed in our runtime environment at the end of this document. Note that Pytorch and Pytorch-geometric can be installed directly on CPU by changing the device in the code.

# How to reproduce our experiments

Every train is already prepared inside PDDLGym. If one wants to train a policy for, say, problem 0 of multi-blocksworld domain, just run:

python main.py --env PDDLEnvBlocksmulti-v0 --num_problem 0

This will train the policy weights and save them inside train_models/PDDLEnvBlocksmulti-v0 with the name "5_actor.pth" and "5_critic.pth". A file "timesteps.txt" is also generated, in which we display the total number of iterations and the total time for training. 

To generate a plan with the trained policy, we run

python main.py --env PDDLEnvBlocksmulti-v0 --num_problem 0 --mode test --max_num 49 --actor_model trained_models/PDDLEnvBlocksmulti-v0/0_actor.pth

And the plan would be saved in the folder plans/PDDLEnvBlocksmulti-v0 as "0.pddl".

The max_num parameter specifies the maximum number of problems that are present for each domain. Values for each domain are as follows:

- PDDLEnvBlocksmulti-v0 (Multi-blocksworld): 49
- PDDLEnvFloortile-v0 (Floortile): 55
- PDDLEnvFree_openstacks-v0 (Free_openstacks): 34
- PDDLEnvLogpure-v0 (Transport): 63
- PDDLEnvOpenstacks-v0 (Openstacks): 34

As the reader can see, domains are addressed in a specific way. 

Note: problems in PDDLGym are addressed in lexical order. That is, problem number 3 is not 3.pddl, but 10.pddl (Because, in lexical order, '0.pddl' < '1.pddl' < '11.pddl').

# Process semantics validator

Here we briefly explain how the process semantics validator works:

The validator is based on a search where each node contains a list where states are stored, labeled with their timestamp, along with the set of actions that can be executed in that state simultaneously. The list of the initial node only contains the initial state, labeled with time zero, and no associated actions.Expanding a node means adding the first action in the plan that does not appear in that node. To add an action a to a node, all possible states are chosen from the list that satisfy the following:

- The preconditions of a are met in that state.
- There are no conflicts between a and the actions that must be executed in that state. A conflict occurs when actions have contradictory effects (one deletes what another adds) or when one action erases some precondition of another one.
- The effects of a do not prevent the preconditions of the actions in the list that must be executed later from being fulfilled.

The search branches according to the possible states where a can be inserted. If the action cannot be inserted into any state, the node is pruned. When a node containing all the actions of the plan is reached (final node), its makespan is taken into account to prune longer solutions. When there are no more nodes left to visit, the search ends and the final node with the best makespan found is returned. If the plan is invalid, the search will terminate without finding a plan and an error will be returned.

The process semantics checker is in file "process_checker", with a makefile attached for reproducibility purposes.

# List of all our packages

# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
alabaster                 0.7.12             pyhd3eb1b0_0  
arrow                     1.2.3           py311h06a4308_1  
astroid                   2.14.2          py311h06a4308_0  
asttokens                 2.0.5              pyhd3eb1b0_0  
atomicwrites              1.4.0                      py_0  
attrs                     23.1.0          py311h06a4308_0  
autopep8                  1.6.0              pyhd3eb1b0_1  
babel                     2.11.0          py311h06a4308_0  
backcall                  0.2.0              pyhd3eb1b0_0  
beautifulsoup4            4.12.2          py311h06a4308_0  
binaryornot               0.4.4              pyhd3eb1b0_1  
black                     23.3.0          py311h06a4308_0  
blas                      1.0                         mkl  
bleach                    4.1.0              pyhd3eb1b0_0  
brotlipy                  0.7.0           py311h5eee18b_1002  
bzip2                     1.0.8                h7b6447c_0  
ca-certificates           2023.08.22           h06a4308_0  
certifi                   2023.7.22       py311h06a4308_0  
cffi                      1.15.1          py311h5eee18b_3  
chardet                   4.0.0           py311h06a4308_1003  
charset-normalizer        2.0.4              pyhd3eb1b0_0  
click                     8.0.4           py311h06a4308_0  
cloudpickle               2.2.1           py311h06a4308_0  
colorama                  0.4.6           py311h06a4308_0  
comm                      0.1.2           py311h06a4308_0  
contourpy                 1.1.1                    pypi_0    pypi
cookiecutter              1.7.3              pyhd3eb1b0_0  
cryptography              41.0.3          py311hdda0065_0  
cycler                    0.12.1                   pypi_0    pypi
cyrus-sasl                2.1.28               h52b45da_1  
dbus                      1.13.18              hb2f20db_0  
debugpy                   1.6.7           py311h6a678d5_0  
decorator                 5.1.1              pyhd3eb1b0_0  
defusedxml                0.7.1              pyhd3eb1b0_0  
diff-match-patch          20200713           pyhd3eb1b0_0  
dill                      0.3.7           py311h06a4308_0  
docstring-to-markdown     0.11            py311h06a4308_0  
docutils                  0.18.1          py311h06a4308_3  
entrypoints               0.4             py311h06a4308_0  
executing                 0.8.3              pyhd3eb1b0_0  
expat                     2.5.0                h6a678d5_0  
filelock                  3.9.0                    pypi_0    pypi
flake8                    6.0.0           py311h06a4308_0  
fontconfig                2.14.1               h4c34cd2_2  
fonttools                 4.43.1                   pypi_0    pypi
freetype                  2.12.1               h4a9f257_0  
fsspec                    2023.4.0                 pypi_0    pypi
giflib                    5.2.1                h5eee18b_3  
glib                      2.69.1               he621ea3_2  
gst-plugins-base          1.14.1               h6a678d5_1  
gstreamer                 1.14.1               h5eee18b_1  
gym                       0.26.2                   pypi_0    pypi
gym-notices               0.0.8                    pypi_0    pypi
icu                       58.2                 he6710b0_3  
idna                      3.4             py311h06a4308_0  
imageio                   2.31.5                   pypi_0    pypi
imagesize                 1.4.1           py311h06a4308_0  
importlib-metadata        6.0.0           py311h06a4308_0  
importlib_metadata        6.0.0                hd3eb1b0_0  
inflection                0.5.1           py311h06a4308_0  
intel-openmp              2023.1.0         hdb19cb5_46305  
intervaltree              3.1.0              pyhd3eb1b0_0  
ipykernel                 6.25.0          py311h92b7b1e_0  
ipython                   8.15.0          py311h06a4308_0  
ipython_genutils          0.2.0              pyhd3eb1b0_1  
isort                     5.9.3              pyhd3eb1b0_0  
jaraco.classes            3.2.1              pyhd3eb1b0_0  
jedi                      0.18.1          py311h06a4308_1  
jeepney                   0.7.1              pyhd3eb1b0_0  
jellyfish                 1.0.1           py311hb02cf49_0  
jinja2                    3.1.2           py311h06a4308_0  
jinja2-time               0.2.0              pyhd3eb1b0_3  
joblib                    1.3.2                    pypi_0    pypi
jpeg                      9e                   h5eee18b_1  
jsonschema                4.17.3          py311h06a4308_0  
jupyter_client            8.1.0           py311h06a4308_0  
jupyter_core              5.3.0           py311h06a4308_0  
jupyterlab_pygments       0.1.2                      py_0  
keyring                   23.13.1         py311h06a4308_0  
kiwisolver                1.4.5                    pypi_0    pypi
krb5                      1.20.1               h143b758_1  
lazy-loader               0.3                      pypi_0    pypi
lazy-object-proxy         1.6.0           py311h5eee18b_0  
ld_impl_linux-64          2.38                 h1181459_1  
lerc                      3.0                  h295c915_0  
libclang                  14.0.6          default_hc6dbbc7_1  
libclang13                14.0.6          default_he11475f_1  
libcups                   2.4.2                h2d74bed_1  
libdeflate                1.17                 h5eee18b_1  
libedit                   3.1.20221030         h5eee18b_0  
libevent                  2.1.12               hdbd6064_1  
libffi                    3.4.4                h6a678d5_0  
libgcc-ng                 11.2.0               h1234567_1  
libgomp                   11.2.0               h1234567_1  
libllvm14                 14.0.6               hdb19cb5_3  
libpng                    1.6.39               h5eee18b_0  
libpq                     12.15                hdbd6064_1  
libsodium                 1.0.18               h7b6447c_0  
libspatialindex           1.9.3                h2531618_0  
libstdcxx-ng              11.2.0               h1234567_1  
libtiff                   4.5.1                h6a678d5_0  
libuuid                   1.41.5               h5eee18b_0  
libwebp                   1.3.2                h11a3e52_0  
libwebp-base              1.3.2                h5eee18b_0  
libxcb                    1.15                 h7f8727e_0  
libxkbcommon              1.0.1                h5eee18b_1  
libxml2                   2.10.4               hcbfbd50_0  
libxslt                   1.1.37               h2085143_0  
lxml                      4.9.3           py311hdbbb534_0  
lz4-c                     1.9.4                h6a678d5_0  
markupsafe                2.1.1           py311h5eee18b_0  
matplotlib                3.8.0                    pypi_0    pypi
matplotlib-inline         0.1.6           py311h06a4308_0  
mccabe                    0.7.0              pyhd3eb1b0_0  
mistune                   0.8.4           py311h5eee18b_1000  
mkl                       2023.1.0         h213fc3f_46343  
mkl-service               2.4.0           py311h5eee18b_1  
mkl_fft                   1.3.8           py311h5eee18b_0  
mkl_random                1.2.4           py311hdb19cb5_0  
more-itertools            8.12.0             pyhd3eb1b0_0  
mpmath                    1.3.0                    pypi_0    pypi
mypy_extensions           1.0.0           py311h06a4308_0  
mysql                     5.7.24               h721c034_2  
nbclient                  0.5.13          py311h06a4308_0  
nbconvert                 6.5.4           py311h06a4308_0  
nbformat                  5.9.2           py311h06a4308_0  
ncurses                   6.4                  h6a678d5_0  
nest-asyncio              1.5.6           py311h06a4308_0  
networkx                  3.1                      pypi_0    pypi
nspr                      4.35                 h6a678d5_0  
nss                       3.89.1               h6a678d5_0  
numpy                     1.26.0          py311h08b1b3b_0  
numpy-base                1.26.0          py311hf175353_0  
numpydoc                  1.5.0           py311h06a4308_0  
openssl                   3.0.12               h7f8727e_0  
packaging                 23.1            py311h06a4308_0  
pandocfilters             1.5.0              pyhd3eb1b0_0  
parso                     0.8.3              pyhd3eb1b0_0  
pathspec                  0.10.3          py311h06a4308_0  
pcre                      8.45                 h295c915_0  
pddlgym                   0.0.5                     dev_0    <develop>
pexpect                   4.8.0              pyhd3eb1b0_3  
pickleshare               0.7.5           pyhd3eb1b0_1003  
pillow                    10.1.0                   pypi_0    pypi
pip                       23.3            py311h06a4308_0  
platformdirs              3.10.0          py311h06a4308_0  
pluggy                    1.0.0           py311h06a4308_1  
ply                       3.11            py311h06a4308_0  
poyo                      0.5.0              pyhd3eb1b0_0  
progressbar               2.5                        py_0    conda-forge
progressbar2              4.2.0           py311h06a4308_0  
prompt-toolkit            3.0.36          py311h06a4308_0  
psutil                    5.9.0           py311h5eee18b_0  
ptyprocess                0.7.0              pyhd3eb1b0_2  
pure_eval                 0.2.2              pyhd3eb1b0_0  
pycodestyle               2.10.0          py311h06a4308_0  
pycparser                 2.21               pyhd3eb1b0_0  
pydocstyle                6.3.0           py311h06a4308_0  
pyflakes                  3.0.1           py311h06a4308_0  
pyg-lib                   0.3.0+pt21cu121          pypi_0    pypi
pygments                  2.15.1          py311h06a4308_1  
pylint                    2.16.2          py311h06a4308_0  
pylint-venv               2.3.0           py311h06a4308_0  
pyls-spyder               0.4.0              pyhd3eb1b0_0  
pyopenssl                 23.2.0          py311h06a4308_0  
pyparsing                 3.1.1                    pypi_0    pypi
pyqt                      5.15.7          py311h6a678d5_0  
pyqt5-sip                 12.11.0         py311h6a678d5_0  
pyqtwebengine             5.15.7          py311h6a678d5_0  
pyrsistent                0.18.0          py311h5eee18b_0  
pysocks                   1.7.1           py311h06a4308_0  
python                    3.11.5               h955ad1f_0  
python-dateutil           2.8.2              pyhd3eb1b0_0  
python-fastjsonschema     2.16.2          py311h06a4308_0  
python-lsp-black          1.2.1           py311h06a4308_0  
python-lsp-jsonrpc        1.0.0              pyhd3eb1b0_0  
python-lsp-server         1.7.2           py311h06a4308_0  
python-slugify            5.0.2              pyhd3eb1b0_0  
python-utils              3.3.3           py311h06a4308_0  
pytoolconfig              1.2.5           py311h06a4308_1  
pytz                      2023.3.post1    py311h06a4308_0  
pyxdg                     0.27               pyhd3eb1b0_0  
pyyaml                    6.0             py311h5eee18b_1  
pyzmq                     25.1.0          py311h6a678d5_0  
qdarkstyle                3.0.2              pyhd3eb1b0_0  
qstylizer                 0.2.2           py311h06a4308_0  
qt-main                   5.15.2               h7358343_9  
qt-webengine              5.15.9               h9ab4d14_7  
qtawesome                 1.2.2           py311h06a4308_0  
qtconsole                 5.4.2           py311h06a4308_0  
qtpy                      2.2.0           py311h06a4308_0  
qtwebkit                  5.212                h3fafdc1_5  
readline                  8.2                  h5eee18b_0  
requests                  2.31.0          py311h06a4308_0  
rope                      1.7.0           py311h06a4308_0  
rtree                     1.0.1           py311h06a4308_0  
scikit-image              0.22.0                   pypi_0    pypi
scikit-learn              1.3.2                    pypi_0    pypi
scipy                     1.11.3                   pypi_0    pypi
secretstorage             3.3.1           py311h06a4308_1  
setuptools                68.0.0          py311h06a4308_0  
sip                       6.6.2           py311h6a678d5_0  
six                       1.16.0             pyhd3eb1b0_1  
snowballstemmer           2.2.0              pyhd3eb1b0_0  
sortedcontainers          2.4.0              pyhd3eb1b0_0  
soupsieve                 2.5             py311h06a4308_0  
sphinx                    5.0.2           py311h06a4308_0  
sphinxcontrib-applehelp   1.0.2              pyhd3eb1b0_0  
sphinxcontrib-devhelp     1.0.2              pyhd3eb1b0_0  
sphinxcontrib-htmlhelp    2.0.0              pyhd3eb1b0_0  
sphinxcontrib-jsmath      1.0.1              pyhd3eb1b0_0  
sphinxcontrib-qthelp      1.0.3              pyhd3eb1b0_0  
sphinxcontrib-serializinghtml 1.1.5              pyhd3eb1b0_0  
spyder                    5.4.3           py311h06a4308_1  
spyder-kernels            2.4.4           py311h06a4308_0  
sqlite                    3.41.2               h5eee18b_0  
stack_data                0.2.0              pyhd3eb1b0_0  
sympy                     1.12                     pypi_0    pypi
tbb                       2021.8.0             hdb19cb5_0  
text-unidecode            1.3                pyhd3eb1b0_0  
textdistance              4.2.1              pyhd3eb1b0_0  
threadpoolctl             3.2.0                    pypi_0    pypi
three-merge               0.1.1              pyhd3eb1b0_0  
tifffile                  2023.9.26                pypi_0    pypi
tinycss2                  1.2.1           py311h06a4308_0  
tk                        8.6.12               h1ccaba5_0  
toml                      0.10.2             pyhd3eb1b0_0  
tomlkit                   0.11.1          py311h06a4308_0  
torch                     2.1.0+cu121              pypi_0    pypi
torch-cluster             1.6.3+pt21cu121          pypi_0    pypi
torch-geometric           2.4.0                    pypi_0    pypi
torch-scatter             2.1.2+pt21cu121          pypi_0    pypi
torch-sparse              0.6.18+pt21cu121          pypi_0    pypi
torch-spline-conv         1.2.2+pt21cu121          pypi_0    pypi
torchaudio                2.1.0+cu121              pypi_0    pypi
torchvision               0.16.0+cu121             pypi_0    pypi
tornado                   6.3.3           py311h5eee18b_0  
tqdm                      4.66.1                   pypi_0    pypi
traitlets                 5.7.1           py311h06a4308_0  
triton                    2.1.0                    pypi_0    pypi
typing-extensions         4.4.0                    pypi_0    pypi
tzdata                    2023c                h04d1e81_0  
ujson                     5.4.0           py311h6a678d5_0  
unidecode                 1.2.0              pyhd3eb1b0_0  
urllib3                   1.26.16         py311h06a4308_0  
watchdog                  2.1.6           py311h06a4308_0  
wcwidth                   0.2.5              pyhd3eb1b0_0  
webencodings              0.5.1           py311h06a4308_1  
whatthepatch              1.0.2           py311h06a4308_0  
wheel                     0.37.1             pyhd3eb1b0_0  
wrapt                     1.14.1          py311h5eee18b_0  
wurlitzer                 3.0.2           py311h06a4308_0  
xz                        5.4.2                h5eee18b_0  
yaml                      0.2.5                h7b6447c_0  
yapf                      0.31.0             pyhd3eb1b0_0  
zeromq                    4.3.4                h2531618_0  
zipp                      3.11.0          py311h06a4308_0  
zlib                      1.2.13               h5eee18b_0  
zstd                      1.5.5                hc292b87_0