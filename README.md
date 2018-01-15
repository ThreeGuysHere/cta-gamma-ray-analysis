# cta-gamma-ray-analysis

1. Install all the libraries listed in `requirements.txt`
2. Run `run.sh`
3. Keep focus on displayed images
4. You can change:
	- **Adaptive Threshold**
		- **blockSize**, Size of a pixel neighbourhood that is used to calculate a threshold value for the pixel
		- **const**, Constant subtracted from the mean or weighted mean
	- **Filters**
		- **number of median filter iterations**
		- **median filter kernel size**
		- **number of gaussian filter iterations**
		- **gaussian filter kernel size**
	- **Local Linear Stretching**
		- **stretch kernel size**
		- **stretch step size**: step size at every sliding window iteration
		- **stretch min bins**: minimum value within which the current window is normalized
		
		
**Legenda**:
Type the following keys to change mode:
- `t`	adaptive threshold
- `f`	gaussian & median filter
- `l`	local stretch

then type `[1 .. 4]` to select the parameter and use `a` and `d` to change it.

Then you can type:

- `r`	to print results 
- `v`	to print current values

- `w` and `s` to change map

- `enter` to print legenda
- `esc`	to quit

	
