# reverse-undistortion
Reversing undistortion step applied in ISP pipelines (mapping image locations from JPEG to RAW annotations)

# Project TODO list: 
- [ ] Decide about M.1.4. Additional downscaling? 
- [ ] Problem description (README)
- [ ] Visualization (Problem descr.) 
- [ ] Stats reader
- [ ] Config reading
- [ ] Method 
- [ ] Metrics? 

# Dataset details
| Atribute | Sony | Nikon |
|---|---|---|
| Resolution | 5472x3648 | 3936x2624 |
| gcd(H,W) | 1824 | 1312 |
| RAW Resolution | 5504x3672 | 3968x2640 | 
| RAW gcd(H,W) | 8 | 16 |

# Method
Input: images @ focal length FL.
1. Prepare images:
   1. RAW: 1->4 ch. 
   2. RAW: RGGB->avg(G,G) \[approx. illumination\]
   3. JPEG: Downscale JPEG 4 times (D)
   4. Additional downscaling? 
2. For each image @ FL:
   1. Estimate homography aligning JPEG->RAW (g)
3. G = mean of all mappings g
4. For each image @ FL:
   1. JPEG' = G(JPEG)
   2. l <- Slide window (transaltion T) and estimate homography aligning each patch G(JPEG')->RAW 
5. L = mean of all mappings l
6. Compose mappings: (L+T) o G o D o R