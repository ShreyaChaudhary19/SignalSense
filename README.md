# SignalSense 🚦
## Near-Miss-Aware Adaptive Traffic Signal Control Using Reinforcement Learning

SignalSense is an academic project exploring reinforcement learning for adaptive traffic signal control using SUMO (Simulation of Urban Mobility). It focuses on studying traffic efficiency and safety-related indicators, including near-miss events, in a simulated traffic environment.

## Problem Statement

Conventional traffic signals may use fixed timings that do not adapt effectively to changing traffic demand. SignalSense explores how reinforcement learning and traffic simulation can help investigate adaptive signal-control strategies and traffic safety.

## Objectives

- Explore reinforcement learning for adaptive traffic signal control.
- Simulate road networks and vehicle movements using SUMO.
- Analyze traffic congestion and traffic-control performance.
- Investigate safety-related indicators such as near-miss events and hard braking.
- Compare experimental results using Python-based analysis and visualizations.

## Technology Stack

- **Python** — programming and analysis
- **SUMO / SUMO-GUI** — traffic simulation
- **TraCI** — Python interface for SUMO
- **SUMOlib** — SUMO utilities
- **PyTorch** — machine learning development
- **NumPy** — numerical computation
- **Matplotlib** — visualization
- **XML** — network, route, and simulation configuration files

## Project Structure

```text
SignalSense/
├── maps/                   # Road networks and route files
├── maps_images/             # Network images
├── models/                  # Saved model artifacts
├── plots/                   # Experimental plots
├── safety_results/          # Safety analysis outputs
├── trafficlight_2/          # Traffic-light-related Arduino code
├── configuration.sumocfg    # SUMO configuration
├── train.py                 # Training script
├── compare_models.py        # Model comparison
├── compare_models_v2.py     # Additional comparison
├── near_miss.py             # Near-miss analysis
├── safety_metrics.py        # Safety metric calculations
├── evaluate_safety.py       # Safety evaluation
├── check_congestion.py      # Congestion analysis
├── test_safety_metrics.py   # Safety metric tests
├── requirements.txt         # Python dependencies
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ShreyaChaudhary19/SignalSense.git
cd SignalSense
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install SUMO

Install SUMO separately and ensure its executable directory is available in your system PATH.

Verify the installation:

```powershell
sumo --version
sumo-gui --version
```

## Running the Simulation

From the project root, launch the SUMO graphical interface:

```powershell
sumo-gui -c ".\configuration.sumocfg"
```

To limit the simulation to 200 simulation-time units:

```powershell
sumo-gui -c ".\configuration.sumocfg" --end 200
```

## Running the Python Scripts

Activate your virtual environment, then run the relevant script from the project directory.

For example:

```powershell
python train.py
```

Other scripts include `near_miss.py`, `safety_metrics.py`, `evaluate_safety.py`, and `check_congestion.py`. Check each script's implementation and configuration before execution.

## Reinforcement Learning Concept

Reinforcement learning involves an agent interacting with an environment and learning from rewards or penalties.

In traffic signal control, the general components are:

- **Agent:** Traffic signal controller
- **Environment:** SUMO traffic simulation
- **State:** Available traffic conditions
- **Action:** A permitted signal-control decision
- **Reward:** Feedback defined by the implemented learning algorithm

The specific algorithm, state representation, action space, and reward function should be verified from the project source code.

## Safety Analysis

The repository includes scripts for near-miss analysis and safety-related evaluation. These can be used to investigate vehicle interactions and indicators such as time-to-collision observations and hard-braking events, depending on the implementation.

Simulation-based indicators do not guarantee real-world safety. Their interpretation depends on the detection methodology, thresholds, and simulation assumptions.

## Evaluation

The project includes scripts and outputs for analyzing traffic-control experiments. Meaningful comparisons should use consistent simulation settings, traffic demand, and evaluation criteria.

Actual performance claims should be based on recorded experimental results rather than assumptions.

## Future Scope

- Improve safety-aware reward design.
- Compare reinforcement learning approaches with baseline signal-control strategies.
- Test across more traffic networks and demand patterns.
- Add a dashboard for simulation metrics and visualizations.
- Evaluate results across multiple random seeds.
- Investigate real-time data integration and further validation.

## Limitations

- Results depend on the simulated traffic environment.
- Safety metrics depend on the definitions and thresholds used.
- Simulation performance does not establish real-world effectiveness.
- Real traffic deployment would require extensive testing and safety validation.

## Author

**Shreya Chaudhary**  
B.Tech — Artificial Intelligence and Data Science  
Dr. D. Y. Patil School of Science and Technology, Thathwade

GitHub: [ShreyaChaudhary19](https://github.com/ShreyaChaudhary19)

## License

No license has been specified yet. Add an appropriate license file if you intend to distribute the project for reuse.

---

*SignalSense is an academic project exploring reinforcement learning, traffic simulation, and safety-aware traffic management.*
