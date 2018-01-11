# cta-gamma-ray-analysis

1. Run `test/blobtest.py`
2. Keep focus on displayed images
3. You can change:
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

- `t`	adaptive threshold
- `f`	gaussian & median filter
- `s`	local stretch

- `r`	print results
- `v`	print current values

- use `up-arrow` and `down-arrow` to change map
- `esc`	quit

	
