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

## Arguments for Class3D/InitialModel Selection (All required)

```
-F: Class3D MRC files to be examine, separated by space
-G: The GPU ID to use for the computation, use comma to seperate multiple GPUs
-J: The Job Name
```

## Example for Class3D/InitialModel Selection

```
python main.py -F ./Class3D/job052/class1.mrc ./Class3D/job052/class2.mrc ./Class3D/job052/class3.mrc -G 0,1,2 -J job052_select
```

## Arguments for Auto Contouring

```
-i: Input MRC map file to determine the contour
-o: Output folder to store all the files
-p: Plot all components (Optional, False by default)
-n: Number of intializations (Optional, 3 by default)
```

## Example for GMM Auto Contouring for Rough Masking

```
python contour.py -i ./Class3D/job052/class1.mrc -o ./output_folder -p
```


## Example for CryoREAD Auto Refinement Masking

```
python contour.py -i ./Class3D/job052/class1.mrc -o ./output_folder -p
```