%
% SATF: Shape-Adaptive Tensor Factorization Model DEMO.
%        Version: 1.0
%        Date   : Apr 2021
%
%    This demo shows the SATF method for hyperspectral image dimensionality reduction and classification.
%
%    IP_main.m ....... A man function implementing the SATF model for Indian Pines data sets.
%    PU_main.m ....... A man function implementing the SATF model for University of Pavia data sets.
%    normcols.m .......A function for normalization.
%
%    /data ................ The folder contains the IP and PU data sets.
%    /LASIP_Image_Restoration_DemoBox_v113 .. The folder contains the Anisotropic Nonparametric Image Restoration DemoBox.
%    /LORSAL ...............The folder contains the LORSAL algorithm.
%    /NFEA .................The folder contains tensor_toolbox (for functions of tensor, fmt, rankingFisher).
%    /SA-DCT_Demobox_v143...The folder contains Pointwise Shape-Adaptive DCT Demobox (for shape-adaptive method). 
%    /tensorlab_2016-03-28..The folder contains Tensorlab Demos Release 3.0 (for functions of mlsvd). 
%
%   --------------------------------------
%   Note: Required toolbox/functions are covered
%   --------------------------------------
%   1. LASIP_Image_Restoration_DemoBox_v113: https://www.cs.tut.fi/~lasip/2D/
%   2. LORSAL: http://www.lx.it.pt/~jun/demos.html
%   3. NFEA: https://faculty.skoltech.ru/people/anhhuyphan
%   4. SA-DCT_Demobox_v143: https://www.cs.tut.fi/~foi/SA-DCT/
%   5. tensorlab_2016-03-28: https://www.tensorlab.net/
%   -- Please cite the original implementation when appropriate.
%   --------------------------------------
%   Cite:
%   --------------------------------------
%
%   [1]Z. Xue, S. Yang, M. Zhang. Shape-Adaptive Tensor Factorization Model for Dimensionality Reduction of Hyperspectral Images[J]. IEEE Access, 2019, 7: 115160-115170.
%
%   --------------------------------------
%   Copyright & Disclaimer
%   --------------------------------------
%
%   The programs contained in this package are granted free of charge for
%   research and education purposes only. 
%
%   Copyright (c) 2021 by Zhaohui Xue
%   zhaohui.xue@hhu.edu.cn
%   https://sites.google.com/site/zhaohuixuers/
