# bayesian-matting
Python implementation of Yung-Yu Chuang, Brian Curless, David H. Salesin, and Richard Szeliski. A Bayesian Approach to Digital Matting. In Proceedings of IEEE Computer Vision and Pattern Recognition (CVPR 2001), Vol. II, 264-271, December 2001


### Requirements
- python 3.5+ (Though it should run on 2.7 with some minor tweaks)
- scipy
- numpy
- numba > 0.30.1 (Not neccesary, but does give a 5x speedup)
- matplotlib
- opencv

### Running the demo
- 'python bayesian_matting.py'
-  sigma (σ) fall off of gaussian weighting to local window
-  N size of window to construct local fg/bg clusters from
-  minN minimum number of known pixels in local window to proceed


### Results
<img alt="Original image" src="https://github.com/MarcoForte/bayesian-matting/blob/master/gandalf.png" width="250">
<img alt="Trimap image" src="https://github.com/MarcoForte/bayesian-matting/blob/master/gandalfTrimap.png" width="250">
<img alt="Result" src="https://github.com/MarcoForte/bayesian-matting/blob/master/gandalfAlpha.png" width="250">



### More Information

For more information see the orginal project website http://grail.cs.washington.edu/projects/digital-matting/image-matting/
This implementation was mostly adapted from Michael Rubinsteins matlab code here, 
http://www1.idc.ac.il/toky/CompPhoto-09/Projects/Stud_projects/Miki/index.html
http://people.csail.mit.edu/mrub/code/bayesmat.zip

### Disclaimer

The code is free for academic/research purpose. Use at your own risk and we are not responsible for any loss resulting from this code. Feel free to submit pull request for bug fixes.

### Contact 
[Marco Forte](https://marcoforte.github.io/) (fortem@tcd.ie)  

#### Original authors:  
[Yung-Yu Chuang](http://www.cs.washington.edu/homes/cyy) 
[Brian Curless](http://www.cs.washington.edu/homes/curless)
[David Salesin](http://www.cs.washington.edu/homes/salesin)
[Richard Szeliski](http://www.research.microsoft.com/~szeliski)
