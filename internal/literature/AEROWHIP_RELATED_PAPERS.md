# AeroWhip: related-paper reading list

Checked 11 September 2026. This is a focused inventory of 42 papers from the project's active literature notes and additional directly relevant whip-control sources, not a systematic or exhaustive survey of all robotics literature.

Each title links to a primary paper, proceedings page, author project, or institutional publication record. Summaries are paraphrases. Connections and reading priorities are our analysis, based on AeroWhip's current plan: retain M0, collect new M1/M2 data, update an effective loaded-UAV response model and a distributed cable model, replan offline, and compare continuous task error and held-out prediction error. These connections are not claims that the proposed experiment has already succeeded.

Read entries 1–8 first. The 2020/2021/2023 Nah papers form a related sequence. IRP's RSS paper and journal extension are listed once. Workshop papers and preprints are labeled; an arXiv date is not necessarily the final publication year. The FLARE entry uses the verified author-manuscript dates rather than assuming final journal metadata. Source access varied; this is a bibliographic/abstract-and-method-level check, not a new full-text review of every paper.

## Read first

### 1. [Wiggle and Go! System Identification for Zero-Shot Dynamic Rope Manipulation](https://arxiv.org/abs/2604.22102)

**Authors:** Arthur Jakobsson; Abhinav Mahajan; Karthik Pullalarevu; Krishna Suresh; Yunchao Yao; Yuemin Mao; Bardienus Duisterhof; Shahram Najam Syed; Jeffrey Ichnowski.

**Publication:** 2026, arXiv; reviewed v2.

**Summary:** Identifies rope parameters from a short exploratory motion, then uses them to optimize goal-directed manipulation.

**Connection to AeroWhip:** One of our closest comparisons: identification enables targeted rope motion. AeroWhip instead studies repeated model refinement with a flying actuator; improvement must be demonstrated experimentally.

### 2. [Iterative Residual Policy: for Goal-Conditioned Dynamic Manipulation of Deformable Objects](https://arxiv.org/abs/2203.00663)

**Authors:** Cheng Chi; Benjamin Burchfiel; Eric Cousineau; Siyuan Feng; Shuran Song.

**Publication:** RSS 2022; later IJRR extension.

**Summary:** Learns how action changes alter a previously observed trajectory, enabling improved rope strikes and cloth swings over successive attempts.

**Connection to AeroWhip:** Direct precedent for between-trial improvement. Explain that we update an explicit forward model and replan; IRP primarily predicts the effects of action corrections.

### 3. [Aerial Robots Carrying Flexible Cables: Dynamic Shape Optimal Control via Spectral Method Model](https://arxiv.org/abs/2403.17565)

**Authors:** Yaolei Shen; Antonio Franchi; Chiara Gabellieri.

**Publication:** IEEE Transactions on Robotics, 2025.

**Summary:** Reduces continuum cable dynamics to a compact model and uses predictive feedback control for a quadrotor carrying a flexible cable; includes physical experiments.

**Connection to AeroWhip:** A central aerial comparison. Flexible-cable modeling and control from a UAV are established; our specific task is free-tip whipping with model updates between executions.

### 4. [Manipulation of Elasto-Flexible Cables with Single or Multiple UAVs](https://arxiv.org/abs/2503.04304)

**Authors:** Chiara Gabellieri; Lars Teeuwen; Yaolei Shen; Antonio Franchi.

**Publication:** IROS 2025.

**Summary:** Models extensible cables as interconnected masses and springs, develops trajectory-generation properties, and experimentally validates cable modeling and feedback.

**Connection to AeroWhip:** Supports the aerial-cable context and the distinction between cable-model accuracy and actual vehicle tracking. We need full command-to-tip prediction, alongside measured-attachment diagnostics.

### 5. [Learning Dynamic Rope Manipulation Using Task-Level Iterative Learning Control](https://flying-knots.github.io/)

**Authors:** Krishna Suresh; Chris Atkeson.

**Publication:** RSS 2026, author project confirmation.

**Summary:** Improves a robot's flying-knot motion by mapping rope errors at a critical task event into next-trial command updates.

**Connection to AeroWhip:** A close alternative to model refinement: task error can improve through action correction. Our claim should concern the tested model-and-replanning procedure, without assuming broader model reuse.

### 6. [Real2Sim2Real: Self-Supervised Learning of Physical Single-Step Dynamic Actions for Planar Robot Casting](https://ieor.berkeley.edu/publication/real2sim2real-self-supervised-learning-of-physical-single-step-dynamic-actions-for-planar-robot-casting/)

**Authors:** Vincent Lim; Huang Huang; Lawrence Yunliang Chen; Jonathan Wang; Jeffrey Ichnowski; Daniel Seita; Michael Laskey; Ken Goldberg.

**Publication:** ICRA 2022.

**Summary:** Tunes a cable simulator using real trajectories, generates simulated examples, and learns cable-casting actions from simulated and physical data.

**Connection to AeroWhip:** Direct precedent for the real-data-to-simulation-to-action loop. Our embodiment and 3D transient whipping task differ from planar cable casting.

### 7. [Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects](https://proceedings.mlr.press/v270/chen25d.html)

**Authors:** Yizhou Chen; Yiting Zhang; Zachary Brei; Tiancheng Zhang; Yuzhen Chen; Julie Wu; Ram Vasudevan.

**Publication:** DEFORM; CoRL 2024, proceedings 2025.

**Summary:** Combines differentiable elastic-rod mechanics with learning for dynamic prediction, perception, planning, and control of cables.

**Connection to AeroWhip:** The closest modeling foundation for combining rod mechanics with learned corrections. That combination is prior art; our contribution must come from the aerial system and its measured refinement.

### 8. [Learning to Throw: Agile and Accurate Cable-Suspended Payload Delivery with a Quadrotor](https://arxiv.org/abs/2606.27603)

**Authors:** Yifan Zhai; Elia Raimondi; Yunfan Ren; Ismail Geles; Yannick Armati; Jiaxu Xing; Davide Scaramuzza.

**Publication:** 2026, arXiv preprint.

**Summary:** Uses a coupled quadrotor–rope–payload simulator to train policies for targeted payload release, including hardware deployment.

**Connection to AeroWhip:** A close agile-aerial comparison. Their task releases a payload; ours targets a retained cable tip. Their force-coupled model should not be conflated with our effective response-model cascade.

## Whip motion, control, and repeated manipulation

### 9. [A Learning-based Control Framework for Fast and Accurate Manipulation of a Flexible Object](https://link.springer.com/article/10.1007/s42235-024-00534-2)

**Authors:** Junyi Wang; Xiaofeng Xiong; Silvia Tolu; Stanislav N. Gorb.

**Publication:** Journal of Bionic Engineering, 2024.

**Summary:** Combines learned or optimized whip motions with adaptive impedance control, visual feedback, and a sim-to-real procedure.

**Connection to AeroWhip:** A direct robotic-whipping reference that also treats execution tracking. It helps motivate why a good planned motion still needs an accurate actuator-response model.

### 10. [Deep Reinforcement Learning of Robotic Manipulation for Whip Targeting](https://findresearcher.sdu.dk/ws/files/290228125/Deep_reinforcement_learning_of_robotic_manipulation_for_whip_targeting.pdf)

**Authors:** Xiang Bai; Junyi Wang; Xiaofeng Xiong; Evangelos Boukas.

**Publication:** SIMPAR 2025; simulation study.

**Summary:** Compares reinforcement-learning approaches and rewards for a simulated robot arm targeting with a whip.

**Connection to AeroWhip:** Relevant to our PPO exploration and target-distance objectives. Its algorithm ranking does not establish which planner is best for our UAV or physical experiment.

### 11. [Learning to manipulate a whip with simple primitive actions – A simulation study](https://doi.org/10.1016/j.isci.2023.107395)

**Authors:** Moses C. Nah; Aleksei Krotov; Marta Russo; Dagmar Sternad; Neville Hogan.

**Publication:** iScience, 2023.

**Summary:** Optimizes a small set of smooth arm-motion parameters to manipulate a high-dimensional whip, and studies sensitivity to the motion parameters.

**Connection to AeroWhip:** Supports compact trajectory representations and attention to timing. Our fitting and real-flight evaluation must add evidence beyond successful simulated motion optimization.


# not used, no conference
### 12. [Human-Inspired Robot Whip Manipulation: Preparatory Actions Increase Range and Reduce Control Effort](https://deformable-workshop.github.io/icra2025/spotlight/01_01_05_Edraki_Human.pdf)

**Authors:** Mahdiar Edraki; Silvia Buscaglione; Rakshith Lokesh; John Peter Whitney; Alireza Ramezani; Dagmar Sternad.

**Publication:** ICRA 2025 deformable-manipulation workshop.

**Summary:** Investigates preparatory actions before a robot whip strike and their effect on range and control effort.

**Connection to AeroWhip:** Motivates documenting our launch and preparatory motion. Preparation is an experimental condition that must remain comparable across M0 and M2.

### 13. [Human-Inspired Control of a Whip: Preparatory Movements Improve Hitting a Target](https://pmc.ncbi.nlm.nih.gov/articles/PMC11715529/)

**Authors:** Mahdiar Edraki; Rakshith Lokesh; Aleksei Krotov; Alireza Ramezani; Dagmar Sternad.

**Publication:** BioRob 2024.

**Summary:** Compares striking alone with preparation followed by striking in a human-inspired whip model.

**Connection to AeroWhip:** Useful background for the fold-and-cast motion and initial-state sensitivity. This is a distinct paper from the later workshop article.

### 14. [Motor control beyond reach—how humans hit a target with a whip](https://pmc.ncbi.nlm.nih.gov/articles/PMC9533004/)

**Authors:** Aleksei Krotov; Marta Russo; Moses C. Nah; Neville Hogan; Dagmar Sternad.

**Publication:** Royal Society Open Science, 2022.

**Summary:** Measures human hand and whip motion during target striking to study coordination and whip dynamics.

**Connection to AeroWhip:** Provides physical motivation for exploiting distributed cable motion. It also makes repeated trials and motion variability relevant to our experimental interpretation.

### 15. [Sonic-speed manipulation of a bull whip using a robot manipulator](https://ishikawa-vision.org/fusion/SonicSpeedManipulation/index-e.html)

**Authors:** Yuji Yamakawa; Kazunori Odani; Masatoshi Ishikawa.

**Publication:** IEEE/ASME AIM 2016.

**Summary:** Uses human-inspired trajectory planning, a whip model, and robot experiments to investigate extremely fast whip-tip motion.

**Connection to AeroWhip:** An early robotic-whip precedent. High-speed whipping itself is established; our current claim concerns targeting error and iterative model refinement, not sonic speed.

### 16. [Dynamic Primitives Facilitate Manipulating a Whip](https://doi.org/10.1109/BioRob49111.2020.9224399)

**Authors:** Moses C. Nah; Aleksei Krotov; Marta Russo; Dagmar Sternad; Neville Hogan.

**Publication:** BioRob 2020.

**Summary:** Studies how simple motion primitives and impedance simplify control of a many-link whip.

**Connection to AeroWhip:** Historical foundation for low-dimensional motion search. Read alongside the later 2021 and 2023 papers rather than treating them as unrelated lines of work.

### 17. [Manipulating a Whip in 3D via Dynamic Primitives](https://par.nsf.gov/servlets/purl/10381925)

**Authors:** Moses C. Nah; Aleksei Krotov; Marta Russo; Dagmar Sternad; Neville Hogan.

**Publication:** IROS 2021.

**Summary:** Demonstrates simulated three-dimensional whip targeting through parameterized primitive arm movements.

**Connection to AeroWhip:** A specific predecessor for 3D targeting with a compact command description. It predates our aerial embodiment and between-flight model updates.

### 18. [Online Impedance Adaptation Facilitates Manipulating a Whip](https://findresearcher.sdu.dk/ws/portalfiles/portal/196381933/Online_Impedance_Adaptation_Facilitates_Manipulating_a_Whip.pdf)

**Authors:** Xiaofeng Xiong; Moses C. Nah; Aleksei Krotov; Dagmar Sternad.

**Publication:** IROS 2021.

**Summary:** Adapts joint impedance during simulated whip motion to improve trajectory tracking and optimization behavior.

**Connection to AeroWhip:** Shows that execution dynamics already matter in whip manipulation. Our aircraft response identification is different from online impedance adaptation.

### 19. [Dynamic Manipulation of Deformable Objects With Implicit Integration](https://www.research-collection.ethz.ch/handle/20.500.11850/479918)

**Authors:** Simon Zimmermann; Roi Poranne; Stelian Coros.

**Publication:** IEEE Robotics and Automation Letters, 2021.

**Summary:** Develops trajectory optimization using derivatives through implicit deformable-object simulation, with simulated and physical demonstrations.

**Connection to AeroWhip:** Foundational precedent for model-based dynamic deformable manipulation. Our sampling planner and experimental refinement loop use a different optimization route.

## doubt
### 20. [Self-Supervised Learning of Dynamic Planar Manipulation of Free-End Cables](https://arxiv.org/abs/2405.09581)

**Authors:** Jonathan Wang; Huang Huang; Vincent Lim; Harry Zhang; Jeffrey Ichnowski; Daniel Seita; Yunliang Chen; Ken Goldberg.

**Publication:** 2024, arXiv preprint.

**Summary:** Calibrates cable simulation from real experiments and learns actions to place a free cable endpoint at planar targets.

**Connection to AeroWhip:** Another close free-end targeting reference. Its planar setting differs from suspended 3D whipping; its error values cannot be ranked directly against our metrics.

## Aerial control and vehicle-response modeling

### 21. [FLARE: Agile Flights for Quadrotor Cable-Suspended Payload System via Reinforcement Learning](https://arxiv.org/abs/2508.09797)

**Authors:** Dongcheng Cao; Jin Zhou; Xian Wang; Shuo Li.

**Publication:** 2025 preprint, revised 2026.

**Summary:** Trains policies for agile suspended-payload navigation and transfers them to real quadrotor flights.

**Connection to AeroWhip:** Useful aerial learning context. Payload transport and onboard feedback differ from our retained-tip task and frozen maneuver execution.

### 22. [Nonlinear Predictive Control of the Continuum and Hybrid Dynamics of a Suspended Deformable Cable for Aerial Pick and Place](https://arxiv.org/abs/2602.17199)

**Authors:** Antonio Rapuano; Yaolei Shen; Federico Califano; Chiara Gabellieri; Antonio Franchi.

**Publication:** ICRA 2026; simulation evaluation.

**Summary:** Combines continuum cable modeling, reduced-order dynamics, and NMPC for cable regulation and payload attachment/detachment.

**Connection to AeroWhip:** Shows that aerial cable research extends beyond swing suppression. Our comparison should identify the whipping task and physical model-refinement evidence specifically.

### 23. [Data-Driven MPC for Quadrotors](https://arxiv.org/abs/2102.05773)

**Authors:** Guillem Torrente; Elia Kaufmann; Philipp Foehn; Davide Scaramuzza.

**Publication:** IEEE Robotics and Automation Letters, 2021.

**Summary:** Learns aerodynamic effects with Gaussian processes and incorporates them into quadrotor predictive feedback control.

**Connection to AeroWhip:** Supports learning unmodeled aircraft behavior. We model the loaded aircraft's response under its existing controller, so we should not call it an equivalent aerodynamic identification method.

### 24. [NeuroBEM: Hybrid Aerodynamic Quadrotor Model](https://arxiv.org/abs/2106.08015)

**Authors:** Leonard Bauersfeld; Elia Kaufmann; Philipp Foehn; Sihao Sun; Davide Scaramuzza.

**Publication:** RSS 2021.

**Summary:** Combines analytical aerodynamic modeling with learned corrections to improve quadrotor prediction.

**Connection to AeroWhip:** A strong precedent for physics-plus-learning aircraft models. Our model is a task-specific effective response predictor, with a different input and modeling boundary.

### 25. [Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation](https://arxiv.org/abs/2508.21065)

**Authors:** Jiahe Pan; Jiaxu Xing; Rudolf Reiter; Yifan Zhai; Elie Aljalbout; Davide Scaramuzza.

**Publication:** IEEE Robotics and Automation Letters, 2026.

**Summary:** Updates residual quadrotor dynamics and adapts policies through differentiable simulation using real experience.

**Connection to AeroWhip:** Relevant to adaptation claims. Their online policy adaptation differs from our offline model fitting and replanning between flights.

## Cable mechanics and learned prediction

### 26. [Discrete Elastic Rods](https://www.cs.columbia.edu/cg/pdfs/143-rods.pdf)

**Authors:** Miklós Bergou; Max Wardetzky; Stephen Robinson; Basile Audoly; Eitan Grinspun.

**Publication:** ACM Transactions on Graphics / SIGGRAPH, 2008.

**Summary:** Introduces discrete elastic-rod mechanics for the geometry and deformation of slender structures.

**Connection to AeroWhip:** The mechanics foundation for our rod model. Cite it while separately specifying our own attachment, damping, integration, and learned-correction choices.

### 27. [Accurate Simulation and Parameter Identification of Deformable Linear Objects using Discrete Elastic Rods in Generalized Coordinates](https://arxiv.org/abs/2310.00911)

**Authors:** Qi Jing Chen; Timothy Bretl; Quang-Cuong Pham.

**Publication:** IROS 2025.

**Summary:** Integrates DER mechanics into MuJoCo and identifies bending and twisting stiffness from physical objects.

**Connection to AeroWhip:** A direct identification reference. It supports separating identifiable material parameters from quantities that merely absorb modeling or initialization error.

### 28. [Learning Deformable Linear Object Dynamics From a Single Trajectory](https://arxiv.org/abs/2407.03476)

**Authors:** Shamil Mamedov; A. René Geist; Ruan Viljoen; Sebastian Trimpe; Jan Swevers.

**Publication:** 2024 preprint; RA-L publication 2025.

**Summary:** Learns a structured neural dynamics model for agile deformable-object motion using a short recorded trajectory.

**Connection to AeroWhip:** Relevant to limited-data model fitting and prediction generalization. Our small experiment should report held-out prediction, not infer data efficiency from a small dataset alone.

### 29. [DeformX: A Versatile Co-Simulation Framework for Deformable Linear Objects](https://arxiv.org/abs/2606.22116)

**Authors:** Yi Yang; Xiang Fei; Lehong Wang; Chenhao Li; Zilin Dai; Henry Kou; Lu Li; Howie Choset.

**Publication:** 2026; author record lists IROS 2026.

**Summary:** Couples Cosserat-rod physics with Isaac Sim for deformable-object simulation and robot learning, including real rope-motion transfer.

**Connection to AeroWhip:** A close simulator-to-real-rope comparison. Detailed rod simulation and sim-trained manipulation already exist; our focus is the aerial system and repeated refinement.

### 30. [FLASH: Fast Learning via GPU-Accelerated Simulation for High-Fidelity Deformable Manipulation in Minutes](https://arxiv.org/abs/2604.17513)

**Authors:** Siyuan Luo; Bingyang Zhou; Chong Zhang; Xin Liu; Zhenhao Huang; Gang Yang; Zhengtao Han; Xiaotian Hu; Eric Yang; Rymon Yu; Ziqiu Zeng; Fan Shi.

**Publication:** 2026, arXiv preprint.

**Summary:** Develops a GPU-oriented simulation framework for contact-rich deformable manipulation and scalable learning.

**Connection to AeroWhip:** Supporting computational context. GPU use alone is not a contribution; report our measured workload, hardware, and runtime if discussing efficiency.

## Real-to-simulation adaptation and residual modeling

### 31. [A Distributional Treatment of Real2Sim2Real for Object-Centric Agent Adaptation in Vision-Driven Deformable Linear Object Manipulation](https://arxiv.org/abs/2502.18615)

**Authors:** Georgios Kamaras; Subramanian Ramamoorthy.

**Publication:** IEEE Robotics and Automation Letters, 2025; revised author manuscript 2026.

**Summary:** Infers distributions over object parameters from visual observations and uses them to train object-specific policies in randomized simulation.

**Connection to AeroWhip:** A close deformable-object Real2Sim2Real reference. Our staged fitting provides point estimates and learned corrections rather than posterior parameter distributions.

### 32. [Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience](https://arxiv.org/abs/1810.05687)

**Authors:** Yevgen Chebotar; Ankur Handa; Viktor Makoviychuk; Miles Macklin; Jan Issac; Nathan Ratliff; Dieter Fox.

**Publication:** SimOpt; ICRA 2019.

**Summary:** Updates simulator parameter distributions from real rollouts interleaved with policy training.

**Connection to AeroWhip:** Foundational citation for improving simulation using deployment experience. Distinguish their distribution adaptation and policy training from our frozen-model generations and replanning.

### 33. [AdaptSim: Task-Driven Simulation Adaptation for Sim-to-Real Transfer](https://irom-lab.princeton.edu/AdaptSim/)

**Authors:** Allen Z. Ren; Hongkai Dai; Benjamin Burchfiel; Anirudha Majumdar.

**Publication:** CoRL 2023.

**Summary:** Adapts simulation parameters to improve downstream real task performance, even when exact physical matching is unattainable.

**Connection to AeroWhip:** Motivates evaluating prediction error and actual target error separately. Better fitted dynamics do not automatically imply a better executed whip.

### 34. [GenDOM: Generalizable One-shot Deformable Object Manipulation with Parameter-Aware Policy](https://arxiv.org/abs/2309.09051)

**Authors:** So Kuroki; Jiaxian Guo; Tatsuya Matsushima; Takuya Okubo; Masato Kobayashi; Yuya Ikeda; Ryosuke Takanami; Paul Yoo; Yutaka Matsuo; Yusuke Iwasawa.

**Publication:** ICRA 2024.

**Summary:** Identifies deformable-object parameters from one demonstration and conditions a simulation-trained policy on those parameters.

**Connection to AeroWhip:** An alternative to repeatedly fitting and replanning: infer object properties and reuse a conditioned policy. Relevant to any claim of broad object generalization.

### 35. [What Went Wrong? Closing the Sim-to-Real Gap via Differentiable Causal Discovery](https://proceedings.mlr.press/v229/huang23c.html)

**Authors:** Peide Huang; Xilun Zhang; Ziang Cao; Shiqi Liu; Mengdi Xu; Wenhao Ding; Jonathan Francis; Bingqing Chen; Ding Zhao.

**Publication:** COMPASS; CoRL 2023.

**Summary:** Learns relationships between simulator parameters and real/simulated trajectory discrepancies to guide simulator adjustment.

**Connection to AeroWhip:** Supports diagnosing which subsystem or parameter causes error. Our manually structured drone/cable stages do not implement causal discovery.

### 36. [Rapid Adaptation of Particle Dynamics for Generalized Deformable Object Mobile Manipulation](https://arxiv.org/abs/2603.18246)

**Authors:** Bohan Wu; Roberto Martín-Martín; Li Fei-Fei.

**Publication:** RAPiD; ICRA 2026, author record.

**Summary:** Trains policies conditioned on deformable dynamics embeddings, then infers those embeddings from deployable observations.

**Connection to AeroWhip:** Shows another meaning of adaptation: latent-context inference. We should distinguish it from explicitly updating the simulator between flights.

### 37. [Sim-to-Real of Soft Robots with Learned Residual Physics](https://arxiv.org/abs/2402.01086)

**Authors:** Junpeng Gao; Mike Yan Michelis; Andrew Spielberg; Robert K. Katzschmann.

**Publication:** IEEE Robotics and Automation Letters, 2024.

**Summary:** Learns corrective forces for a physical simulator using sparse measured positions of soft robots.

**Connection to AeroWhip:** A direct precedent for learning residual dynamics from motion markers. Our residuals require their own held-out validation and do not inherit this method's results.

### 38. [RAFL: Generalizable Sim-to-Real of Soft Robots with Residual Acceleration Field Learning](https://arxiv.org/abs/2603.22039)

**Authors:** Dong Heon Cho; Boyuan Chen.

**Publication:** 2026, arXiv preprint.

**Summary:** Learns local residual acceleration fields through a differentiable simulator, targeting transfer across soft-body geometries.

**Connection to AeroWhip:** Especially relevant to our acceleration corrections. Cross-geometry transfer is their evaluated objective; our one-cable study should not imply the same generalization.

## Optimization and methodological foundations


## didnt see the conference
### 39. [Model Predictive Path Integral Control using Covariance Variable Importance Sampling](https://arxiv.org/abs/1509.01149)

**Authors:** Grady Williams; Andrew Aldrich; Evangelos Theodorou.

**Publication:** 2015 manuscript.

**Summary:** Introduces a sampling-based path-integral control formulation with parallel trajectory evaluation and importance sampling.

**Connection to AeroWhip:** The foundation for our MPPI-inspired optimizer. Describe our implementation as offline trajectory search, since the exported strike does not run receding-horizon cable feedback.

### 40. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

**Authors:** John Schulman; Filip Wolski; Prafulla Dhariwal; Alec Radford; Oleg Klimov.

**Publication:** 2017 manuscript.

**Summary:** Introduces policy-gradient objectives designed to support stable optimization over repeated minibatch updates.

**Connection to AeroWhip:** Cite if PPO remains in the paper's methods or comparison. It is optional for a manuscript centered on MPPI and the M0–M2 refinement experiment.

### 41. [On the smoothness of nonlinear system identification](https://arxiv.org/abs/1905.00820)

**Authors:** Antônio H. Ribeiro; Koen Tiels; Jack Umenberger; Thomas B. Schön; Luis A. Aguirre.

**Publication:** Automatica, 2020.

**Summary:** Analyzes how simulation horizon can make identification objectives difficult to optimize and studies constrained multiple shooting.

**Connection to AeroWhip:** Useful for explaining fitting windows and optimization sensitivity. Our independently initialized windows are not constrained multiple shooting.

### 42. [Do Differentiable Simulators Give Better Policy Gradients?](https://proceedings.mlr.press/v162/suh22b.html)

**Authors:** Hyung Ju Suh; Max Simchowitz; Kaiqing Zhang; Russ Tedrake.

**Publication:** ICML 2022.

**Summary:** Analyzes when stiffness and discontinuities limit gradient estimators obtained from differentiable simulation.

**Connection to AeroWhip:** Supports checking numerical derivatives and optimization behavior. Differentiability alone does not prove that fitting or control gradients will be useful.

## Using this list in the manuscript

The related-work discussion can focus on three questions: what has already been demonstrated in dynamic free-end rope manipulation; what has been demonstrated with aerial flexible cables; and how real observations have been used to improve dynamics models or task actions. The mechanics and optimizer papers support the methods section. There is no need to cite all 42 papers in a short manuscript.

For our current one-target study, the clearest framing is an aerial whipping system evaluated through successive model refinement and offline replanning, with model-prediction error separated from executed target error. Broader model reuse, cross-object transfer, algorithmic superiority, and real-time adaptation need separate evidence. M0-to-M2 also changes cable-residual capacity, which must be stated when interpreting improvement.

Source inventory was assembled from the existing main-branch HANDOFF.md, docs/paper/ICRA_2027_RELATED_WORK_AND_FRAMING.md, docs/paper/related_papers.bib, and the research-design notes, then cross-checked against the linked external sources. Older local descriptions of force-based PPO or binary success gates were not used to define the current project.
