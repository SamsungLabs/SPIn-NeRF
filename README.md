# SIn-NeRF2NeRF: Editing 3D Scenes with Instructions through Segmentation and Inpainting

![teaser.png](readme%206e8ba03ecbe04d6f9c9358a593d9a079/teaser.png)

We propose the method SIn-NeRF2NeRF, which performs 3D object editing selectively by disentangling it from the background scene.

---

## Implementation Details

We have implemented the entire process from data preparation/preprocessing (2D image multiview segmenta- tion) to applying in2n on the object scene and synthesizing the 3D object scene with the 3D inpainted background scene.

- **2D Image Multiview Segmentation**
    
    We implemented code that allows interactive 2D image multiview segmentation using the Segment Anything Model (SAM)
    

- **Object Scene Reconstruction & Applying in2n**
    
    For the object scene, we train a DSNeRF using an RGBA image set concatenated from the object mask and object image. During this process, we implement the random background color technique borrowed from the instant-ngp code to ensure effective training of the object scene with a transparent background.
    

We develop a code to edit the object scene based on text prompts by porting the in2n pipeline, especially the Iter- ative Dataset Update (IDU) algorithm, from Nerfacto to DSNeRF. Our novelty lies in the manipulation of object scenes, since we modify the in2n framework to get input of RGBA images, as opposed to the conven- tional RGB images.

- **Background Scene Reconstruction**
    
    Using the code provided by SPIn-NeRF, we train the DSNeRF for the inpainted background scene.
    
- **3D NeRF Scene Synthesis**
    
    Since both scenes share the same camera parameters, we sort the sampled points for the same rays generated in the object and background scenes by their depth values.
    
- **Object Transformation**
    
    The process of transforming (scaling, translation, and rotation) a disentangled object scene from the background commences with the utilization of COLMAP to ascertain the coordinates of the 3D object. Subsequently, these coordinates are transformed around the centroid using a transformation factor*s,* thereby acquiring the 3D coordinates of the transformed object. This transformation doesn’t change the actual 3D model but alters how it is perceived from the camera’s viewpoint during the rendering process.
    

---

## Main Results


**Qualitative Results**

[rd.mp4](readme%206e8ba03ecbe04d6f9c9358a593d9a079/rd.mp4)

**Quantitative Results**

We offer quantitative measures to assess how well the edits align with the text and the consistency of subsequent frames in CLIP space.

![스크린샷 2023-11-30 오전 12.20.53.png](readme%206e8ba03ecbe04d6f9c9358a593d9a079/chart.png)

---

## Acknowledgements

In this project, we borrowed code from SPIn- NeRF, SAM, Colmap, instant-ngp, ML-Neuman, Instruct- NeRF2NeRF, etc.

| Reference | Purpose |
| --- | --- |
| SPIn-NeRF | Multiview inpainting |
| Instruct-NeRF2NeRF | Iterative dataset update |
| SAM | Interactive object segmentation and mask generation |
| COLMAP | Get true depth using camera parameters |
| ML-Neuman | Merge two different NeRF scenes |
| Instant-ngp | Random background color |

---

## Quick Start

---

## Reference

[https://instruct-nerf2nerf.github.io](https://instruct-nerf2nerf.github.io/)

[https://spinnerf3d.github.io](https://spinnerf3d.github.io/)
