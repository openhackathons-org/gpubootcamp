[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/gpuhackathons-org/gpubootcamp?include_prereleases)](https://github.com/gpuhackathons-org/gpubootcamp/releases/latest) [![GitHub issues](https://img.shields.io/github/issues/gpuhackathons-org/gpubootcamp)](https://github.com/gpuhackathons-org/gpubootcamp/issues)


#  GPUBootcamp Official Training Materials
GPU Bootcamps are designed to help build confidence in Accelerated Computing and eventually prepare developers to enroll for [Hackathons](http://gpuhackathons.org/)

This repository consists of GPU bootcamp material for HPC, AI and convergence of both:

- [HPC](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/hpc) :: 
The bootcamp content focuses on how to follow the Analyze, Parallelize and Optimize Cycle to write parallel codes using different parallel programming models accelerating HPC simulations.

| Lab      | Description |
| ----------- | ----------- |
| [N-Ways](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/hpc/nways)      | This Bootcamp will cover multiple GPU programming models and choose the one that best fits your needs. The material supports different programming langauges including C ( CUDA C, OpenACC C, OpenMP C, C++ stdpar ),  Fortran ( CUDA Fortran, OpenACC Fortran, OpenMP Fortran, ISO DO CONCURRENT ) Python ( Numba, CuPy )       |
| [OpenACC](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/hpc/openacc)   | The Bootcamp will cover how to write portable parallel program that can run on multicore CPUs and accelerators like GPUs and how to apply incremental parallelization strategies using OpenACC       |
| [Multi GPU Programming Model](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/hpc/multi_gpu_nways)   | This bootcamp will cover scaling applications to multiple GPUs across multiple nodes. Moreover, understanding of the underlying technologies and communication topology will help us utilize high-performance NVIDIA libraries to extract more performance out of the system     |


- [Convergence of HPC and AI](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/hpc_ai) :: 
The bootcamp content focuses on how AI can accelerate HPC simulations by introducing concepts of Deep Neural Networks, including data pre-processing, techniques on how to build, compare and improve accuracy of deep learning models. 

| Lab      | Description |
| ----------- | ----------- |
| [Weather Pattern Recognition](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/hpc_ai/ai_science_climate)      | This Bootcamp will introduce developers to fundamentals of AI and how data driven approach can be applied to Climate/Weather domain |
| [CFD Flow Prediction](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/hpc_ai/ai_science_cfd)      | This Bootcamp will introduce developers to fundamentals of AI and how they can be applied to CFD (Computational Fluid Dynamics) |
| [PINN](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/hpc_ai/ai_science_cfd)      | This Bootcamp will introduce developers to fundamentals of using Physics Informed Neural Network and how they can be applied to different scientific domains using Nvidia SimNet |

- [AI](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/ai)::
The bootcamp content focuses on using popular accelerated AI frameworks and using optimization techniques to get max performance from accelerators like GPU.


| Lab      | Description |
| ----------- | ----------- |
| [Accelerated Intelligent Video Analytics](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/ai/DeepStream) | Learn how Nvidia DeepStream SDK can be used to create optimized Intelligent Video Analytics (IVA) pipeline. Participants will be exposed to the building blocks for creating IVA pipeline followed by profiling exercise to identify hotspots in the pipeline and methods to optimize and get higher throughput       |
| [Accelerated Data Science](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/ai/RAPIDS)   | Learn how RAPIDS suite of open source software libraries gives you the freedom to execute end-to-end data science and analytics pipelines entirely on GPUs. Participants will be exposed to using libraries that can be easily integrated with the daily data science pipeline and accelerate computations for faster execution       |
| [Distributed Deep Learning](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/ai/Distributed_Deep_Learning)   | This bootcamp will introduce participants to fundamentals of Distributed deep learning and give a hands-on experience on methods that can be applied to Deep learning models for faster model training |

# System Requirements
Each lab contains docker and singularity definition files. Follow the readme files inside each on how to build the container and run the labs inside it.

# Contribution
- The repository uses Apache 2.0 license. For more details on folder structure developers may refer to CONTRIBUTING.md file.
- A project template for reference is located at [Template](https://github.com/bharatk-parallel/gpubootcamp-1/tree/nways_md_fortran/misc/jupyter_lab_template/appName)

## Authors and Acknowledgment

See [Contributors](https://github.com/gpuhackathons-org/gpubootcamp/graphs/contributors) for a list of contributors towards this Bootcamp.


# Feature Request or filing issues
- Bootcamp users may request for newer training material or file a bug by filing a github issues
- Please do go through the existing list of issues to get more details of upcoming features and bugs currently being fixed [Issues](https://github.com/gpuhackathons-org/gpubootcamp/issues)

## General Troubleshooting

- All materials developed are tested with latest GPU Architectures (V100, A100). Most labs unless specified explicitly are expected to work even on older GPU architectures and with lesser compute and memory capacity like the one present even in laptops. There will be change in performance results observed based on GPU used. In case you see any issue using the material on other GPU please file an issue in Github mentioning the details of GPU and CUDA Driver version installed.
- The material developed are tested inside container environment like Docker and Singularity. In case the users don't have container environment in the cluster, they can explicitly look at the steps mentioned in the Dockerfile and Singularity scripts and install the dependenciesmanually.
- All bootcamps are jupyter based and by default the Dockerfile and Singularity script runs the jupyter notebook at port 8888. In a munti-tenancy environment the admins are requested to explicitly map the ports to individual users else will result into port conflict issues. We recommend having installations of interactive interface to remote computing resources like [Open OnDemand](https://openondemand.org/) or [JupyterHub](https://jupyter.org/hub) coupled with scheduler (SLURM, Kubernetes etc ) to do these resources mapping automatically. 

## Join OpenACC Community
Please join [OpenACC Slack Channel](https://openacclang.slack.com/messages/openaccusergroup).
