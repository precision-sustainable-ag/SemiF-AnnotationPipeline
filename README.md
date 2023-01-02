
We use git subtree to organize this project. This repos contain

After pulling this repo,

# SemiF-AnnotationPipeline
## 1. Pull this remote

```
git clone git@github.com:mkutu/Pipeline.git
```

## 2. Change branchs

Move into the main project
```
cd Pipeline
```

Change to the correct branch. At the time of writting, `develop`
```
git checkout -b develop origin/restruct
```

## 3. Add SemiF-AnnotationPipeline as a remote

Add SemiF-AnnotationPipeline sub-project as a remote. Refer to the remote as `ann_origin` to keep it short.
```bash
git remote add -f ann_origin git@github.com:precision-sustainable-ag/SemiF-AnnotationPipeline.git
```

## 4. Fetch and pull the SemiF-AnnoationPipeline branch
Fetch the correct sub-project branch. At the time of writting, `sfann-hydra-restruct`
```bash
git fetch ann_origin sfann-hydra-restruct
```
```bash
git subtree pull --prefix SemiF-AnnotationPipeline ann_origin sfann-hydra-restruct
```

# SemiF-SyntheticPipeline

The steps are exactly the same starting from #3 except different branch names (at time of writing) and abbreviations are used (`develop` and `synth_origin` respectively). Steps are condensed to just code below.

```
git remote add -f synth_origin git@github.com:precision-sustainable-ag/SemiF-SyntheticPipeline.git
git fetch synth_origin develop
git subtree pull --prefix SemiF-SyntheticPipeline synth_origin develop
```

# Push a subtree repo upstream

`git subtree push --prefix=subtree subtree_origin branch`

For examples,
```
git subtree push --prefix=Semif-SyntheticPipeline synth_origin develop
```


