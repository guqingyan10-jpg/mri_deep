# Method naming convention

| Abbreviation | English name | Definition |
|---|---|---|
| LHFC | Laplacian High-Frequency Feature Concatenation | Laplacian high-frequency features concatenated into the decoder. |
| ABS | Auxiliary Boundary Supervision | Independent boundary supervision weighted by lambda_b. |
| AR-MSC | Adaptive Residual Multi-Scale Context | Residual multi-scale context controlled by one learnable scalar alpha. |
| AFBMS-ResUNet | Adaptive Frequency-Boundary Multi-Scale ResUNet | Complete method: ResUNet + LHFC + ABS + AR-MSC. |

AFBMS-ResUNet uses plain LHFC concatenation (`fusion=concat`). 
It does not use gated concatenation. The learnable scalar alpha belongs 
only to AR-MSC in the V2 model and is not a feature-fusion gate.
