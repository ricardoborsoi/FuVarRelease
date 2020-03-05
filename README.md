#   Super-Resolution for Hyperspectral and Multispectral Image Fusion Accounting for Seasonal Spectral Variability    #

This package contains the authors' implementation of the paper [1].

Most Hyperspectral (HS) and Multispectral (MS) image fusion models assume that both images were acquired under the same conditions. Thus, when the HS and MS images are acquired at different time instants, the presence of seasonal or acquisition (e.g., illumination, atmospheric) variations often impacts negatively the performance of the algorithms. In this work we considered a more flexible model that takes such variability into account, performing consistently well even when significant variations are observed.


The code is implemented in MATLAB and includes:  
-  example2.m                - a demo script comparing the algorithms (Paris image)  
-  ./FuVar/                  - contains the MATLAB files associated with the FuVar algorithm  
-  ./utils/                  - useful functions, metrics and other methods  
-  ./DATA/                   - data files used in the examples  
-  README                    - this file  



## IMPORTANT:
If you use this software please cite the following in any resulting
publication:

    [1] Super-Resolution for Hyperspectral and Multispectral Image Fusion Accounting for Seasonal Spectral Variability
        R.A. Borsoi, T. Imbiriba, J.C.M. Bermudez.
        IEEE Transactions on Image Processing, 2019.



## INSTALLING & RUNNING:
Just start MATLAB and run example1.m or example2.m.


## NOTES:
1.  The GLPHS, HySure, CNMF algorithms were provided by Naoto Yokoya, in the toolbox associated with the paper:
    Naoto Yokoya, Claas Grohnfeldt, Jocelyn Chanussot.
    Hyperspectral and multispectral data fusion: A comparative review of the recent literature.
    IEEE Geoscience and Remote Sensing Magazine, 2017.
    Downloadable [here](https://openremotesensing.net/wp-content/uploads/2017/11/HSMSFusionToolbox.zip). 










