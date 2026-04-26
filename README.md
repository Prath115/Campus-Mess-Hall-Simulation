
***

# 🍔 Campus Mess Hall Queue Dynamics Simulator

A hybrid simulation project combining **Discrete Event Simulation (DES)** and **Agent-Based Modelling (ABM)** to study real-world queue behavior in a campus mess hall. Built with Python, SimPy, and Streamlit, this tool visualizes how academic schedules, human psychology, and social networks impact dining facility bottlenecks.

## 🌟 Key Features

* **Live Streamlit Dashboard:** Watch the campus wake up in real-time. The visualizer animates queue lengths, seating capacities, and serving metrics minute-by-minute.
* **Monte Carlo Analyzer:** Run hundreds of simulated mornings in seconds to extract smooth, mathematically reliable expected values for wait times and arrival counts.
* **Agent-Based Psychology:**
  * **Procrastination Curve:** Student wake-up times follow a dynamic Beta distribution relative to their specific class schedules.
  * **The 80/20 Late Rule:** Probabilistic decision-making where late students decide between skipping breakfast or eating in a rushed state.
* **Network & Social Layer (Homophily):** Simulates "herd behavior" by grouping 80% of students into friend groups based on matching academic branches, causing massive synchronized spikes in the queue.
* **Dynamic Modifiers:** Toggle external factors like **Campus Fatigue**, **Menu Quality**, or a **Post-Cultural Fest Hangover** to see how systemic shocks impact mess hall attendance.
* **Missed Breakfast Matrix:** Automatically calculates and displays a heat-map of exactly which academic batches overslept or missed the mess hall closing time.

## 🛠️ Tech Stack

* **Simulation Engine:** [SimPy](https://simpy.readthedocs.io/)
* **Interactive Frontend:** [Streamlit](https://streamlit.io/)
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Matplotlib

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/mess-hall-simulator.git
   cd mess-hall-simulator
   ```

2. **Install the required dependencies:**
   Make sure you have Python installed. Then, run:
   ```bash
   pip install streamlit simpy pandas matplotlib numpy
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```
   *The interactive dashboard will automatically open in your default web browser.*

## 🎮 How to Use the Dashboard

1. **Set Facility Constraints:** Use the sidebar to adjust the physical limits of the mess hall (Number of Serving Counters, Seating Capacity, Serving Speeds).
2. **Design the Timetable:** Expand the Academic setup in the sidebar to assign 8:00 AM classes, 9:00 AM classes, or no classes to specific branches (MnC, CS, ME, EE).
3. **Tweak the Psychology:** Drag the Alpha and Beta sliders to change the shape of the *Sleep Procrastination Curve* displayed at the top of the app.
4. **Run Live Mode:** Go to the "Live Animation Visualizer" tab and click **Start** to watch a single, animated day unfold.
5. **Run Monte Carlo Mode:** Go to the "Monte Carlo Analyzer" tab, set your desired number of runs, and click **Analyze** to generate a smoothed expected-arrival curve and the Average Missed Breakfast matrix.

## 🧠 Model Assumptions

To maintain mathematical stability, this simulation operates under the following boundaries:
* **Infinite Queue Space:** Students do not balk (leave) due to long lines.
* **Constant Server Efficiency:** Serving staff speed remains constant without fatigue during rush hours.
* **Instant Seating Turnover:** Zero delay for cleaning/clearing plates once a student finishes eating.
* **Rigid Closing Time:** The mess hall doors strictly lock at Minute 120 (9:30 AM).

## 🤝 Acknowledgments

* Designed and authored as a comprehensive study on bottleneck formations and queue theory.
* AI-assisted code generation (Google Gemini) was utilized to construct the SimPy environment syntax, Pandas data-binning logic, and the Streamlit UI wrapper. Core architecture, phasing, and behavioral parameters were manually authored.
