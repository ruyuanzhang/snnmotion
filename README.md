# snnmotion
Spiking Neural Network of Center-Surround Interaction in Motion Perception



# setup

You need to first download and install the forked version of neurogym

```bash
git clone git@github.com:ruyuanzhang/neurogym.git
cd neurogym
pip install -e .
```

Then you need to fetch the 'RNNmotion' branch and merge to a local branch

```bash
git checkout -b RNNmotion # create a local branch
git fetch origin RNNmotion # fetch remote branch RNNmotion
git merge origin/RNNmotion RNNmotion # merge remote branch to local
```

