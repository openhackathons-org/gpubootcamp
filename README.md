#  GPUBootcamp Official Training Materials
GPU Bootcamps are designed to help build confidence in Accelerated Computing and eventually prepare developers to enroll for [Hackathons](http://gpuhackathons.org/)

This repository consists of GPU bootcamp material for HPC, AI and convergence of both:

- [HPC](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/hpc)
The bootcamp content focuses on how to follow the Analyze, Parallelize and Optimize Cycle to write parallel codes using different parallel programming models accelerating HPC simulations.

| Lab      | Description |
| ----------- | ----------- |
| [N-Ways](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/hpc/nways)      | Learn about multiple GPU programming models and choose the one that best fits your needs. The material supports different programmin glangauges including C ( CUDA C, OpenACC C, OpenMP C, C++ stdpar ),  Fortran ( CUDA Fortran, OpenACC Fortran, OpenMP Fortran, ISO DO CONCURRENT       |
| [OpenACC](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/hpc/openacc)   | The lab will cover how to write portable parallel program that can run on multicore CPUs and accelerators like GPUs and how to apply incremental parallelization strategies using OpenACC       |

- [Convergence of HPC and AI](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/hpc_ai)
The bootcamp content focuses on how AI can accelerate HPC simulations by introducing concepts of Deep Neural Networks, including data pre-processing, techniques on how to build, compare and improve accuracy of deep learning models.
| Lab      | Description |
| ----------- | ----------- |
| [Weather Pattern Recognition](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/hpc_ai/ai_science_climate)      | This Bootcamp will introduce developers to fundamentals of AI and how data driven approach can be applied to Climate/Weather domain |
| [CFD Flow Prediction](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/hpc_ai/ai_science_cfd)      | This Bootcamp will introduce developers to fundamentals of AI and how they can be applied to CFD (Computational Fluid Dynamics) |

- [AI](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/ai)
The bootcamp content focuses on using popular accelerated AI frameworks and using optimization techniques to get max performance from accelerators like GPU.

| Lab      | Description |
| ----------- | ----------- |
| [Accelerated Intelligent Video Analytics](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/ai/DeepStream) | Learn how Nvidia DeepStream SDK can be used to create optimized Intelligent Video Analytics (IVA) pipeline. Participants will be exposed to the building blocks for creating IVA pipeline followed by profiling exercise to identify hotspots in the pipeline and methods to optimize and get higher throughput       |
| [Accelerated Data Science](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/ai/RAPIDS)   | Learn how RAPIDS suite of open source software libraries gives you the freedom to execute end-to-end data science and analytics pipelines entirely on GPUs. Participants will be exposed to using libraries that can be easily integrated with the daily data science pipeline and accelerate computations for faster execution       |

# System Requirements
Each lab contains docker and singularity definition files. Follow the readme files inside each on how to build the container and run the labs inside it.

# Contribution
- The repository uses Apache 2.0 license. For more details on folder structure developers may refer to CONTRIBUTING.md file.
- A project template for reference is located at [Template](https://github.com/bharatk-parallel/gpubootcamp-1/tree/nways_md_fortran/misc/jupyter_lab_template/appName)

# Feature Request or filing issues
- Bootcamp users may request for newer training material or file a bug by filing a github issues
- Please do go through the existing list of issues to get more details of upcoming features and bugs currently being fixed [Issues](https://github.com/gpuhackathons-org/gpubootcamp/issues)

## Questions?
Please join [OpenACC Slack Channel](https://openacclang.slack.com/messages/openaccusergroup) for questions.
