# Notices

GravitationalWaveforms.jl  
Copyright (C) 2026 Paul Hopkins

This program is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version. See [LICENSE](LICENSE).

## Upstream code

Parts of this package are Julia ports or machine translations of code from the projects below. The
source comments and `scripts/translate_pyseobnr_*.py` name the routines concerned.

| Upstream | Licence | Used for |
|---|---|---|
| [pySEOBNR](https://git.ligo.org/waveforms/software/pyseobnr) 0.3.7, by S. Ossokine, L. Pompili, D. Mihaylov, A. Ramos Buades, M. Puerrer, H. Estelles, with maintainer R. Enficiaud | GPL-3.0-or-later | SEOBNRv5 Hamiltonian source, fits, flux and PN equations, and the models' dynamics |
| [LALSuite](https://git.ligo.org/lscsoft/lalsuite), Copyright (c) the LALSuite authors | GPL-2.0-or-later | IMRPhenom and post-Newtonian routines and coefficients |
| [ripple](https://github.com/tedwards2412/ripple), Copyright (c) 2022 Adam Coogan, Thomas Edwards | MIT | IMRPhenom routines |
| [qnm](https://github.com/duetosymmetry/qnm), by L. C. Stein | MIT | Quasi-normal-mode data behind the fits in `SEOBNRv5HM_qnm.jl` |

## ripple licence

    MIT License
    
    Copyright (c) [2022] [Adam Coogan, Thomas Edwards]
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## Earlier snapshot

The tag `paper1-run` records the source tree used for the runs of the accompanying paper. Its files
say MIT, but it contains the same upstream-derived code, so that snapshot is also distributed under
GPL-3.0-or-later.
