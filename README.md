# AutoClass3D
A deep learning based tool to automatically select the best reconstructed 3D maps within a group of maps.


## Installation
clone the repository:
```
git clone github.itap.purdue.edu/kiharalab/AutoClass3D
```
create conda environment:
```
conda env create -f environment.yml
```

## Arguments (All required)
```
-F: Path to the folder that contain all the input mrc files
-G: The GPU ID to use for the computation
-J: The Job Name
```

## Example
```
python main.py -F ./Class3D/job052 -G 1 -J job052_select
```